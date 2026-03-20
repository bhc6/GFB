import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer,AutoModelForCausalLM
import argparse
import numpy as np
import wandb
import copy
import random
import heapq
import utils
from dataset_utils import load_all_dataset,dataset_dicts,load_qa_dataset,qa_dicts,load_generation_dataset
# PEFT/TRL removed — using inference-only models
def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_model',type=str,default='google/gemma-3-1b-it')
    parser.add_argument('--agent_model',type=str,default='Qwen/Qwen3-0.6B-Base')
    parser.add_argument('--task',type=str,default='qa')
    parser.add_argument('--dataset',type=str,default='sst2')
    parser.add_argument(
        '--verbalizer',
        type = str,
        nargs = '+',
        default = None
    )
    parser.add_argument('--cache_dir',type=str,default='/mnt/sdb/llm/')
    parser.add_argument('--batch_size',type=int,default=4)
    parser.add_argument('--max_prompt_length',type=int,default=100)
    parser.add_argument('--train_data_per_labels',type=int,default=10)
    parser.add_argument('--num_example',type=int,default=3)
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--meta_prompt',type=str,
                        default = '''I gave a friend an instruction and five inputs. 
                        The friend read the instruction and wrote an output for every one of the inputs.
                        Here are the input-output pairs: \n''')
    parser.add_argument('--prompt_per_example',type=int,default=3)
    parser.add_argument('--update_term',type=int,default=10)
    parser.add_argument('--update_threshold',type=float,default=0.05)   
    parser.add_argument('--num_test_example',type=int,default=20)

    args = parser.parse_args()
    return args

def main():
    
    args = parser_args()
    device= 'cuda:0'
    wandb.init(project='mmlu', 
               config=args,
               name = args.dataset+'meta_change' + args.agent_model + '_' + args.target_model)
    
    
    if args.task == 'classification':
        dataset = load_all_dataset(args.dataset)
        train_dataset = dataset[0]
        test_dataset = dataset[2]
        test_dataset = utils.create_balanced_subset(test_dataset,100)
        if args.verbalizer is None:
            verbalizer = dataset_dicts(args.dataset)
        num_labels = len(verbalizer)
        train_dataset,validation_dataset = utils.create_balanced_subset_and_validation(train_dataset,
                                                                                       args.train_data_per_labels * num_labels,
                                                                                       )
    elif args.task == 'qa':
        dataset = load_qa_dataset(args.dataset)
        train_dataset = dataset[0]
        test_dataset = dataset[2]
        #test_dataset = utils.create_balanced_subset(test_dataset,100)
        if args.verbalizer is None:
            verbalizer = qa_dicts()
        num_labels = len(verbalizer)
        validation_dataset = dataset[4]
    
    elif args.task == 'generation':
        dataset = load_generation_dataset(args.dataset)
        train_dataset = dataset[0]
        test_dataset = dataset[2]
        test_dataset = utils.create_balanced_subset(test_dataset,100)
        verbalizer = None
        validation_dataset = train_dataset
        
    print('train dataset size : ',len(train_dataset))
    print('test dataset size : ',len(test_dataset))
        
    #make dataloader
    test_dataloader = DataLoader(test_dataset,batch_size = 4,shuffle = True)
    train_dataloader = DataLoader(train_dataset,batch_size = 4,shuffle = True)
    
    
    # load agent model (inference-only; RL components removed)
    agent_tokenizer = AutoTokenizer.from_pretrained(args.agent_model,cache_dir = args.cache_dir)
    agent_model = AutoModelForCausalLM.from_pretrained(
        args.agent_model,
        torch_dtype=torch.bfloat16,
        device_map = 'auto',
        cache_dir = args.cache_dir
    )
    agent_tokenizer.pad_token = agent_tokenizer.eos_token
    
    #load target model
    target_tokenizer = AutoTokenizer.from_pretrained(args.target_model,cache_dir = args.cache_dir)
    target_model = AutoModelForCausalLM.from_pretrained(args.target_model,
                                                        cache_dir = args.cache_dir,
                                                        torch_dtype=torch.bfloat16,
                                                        device_map='auto')
    target_model.config.pad_token_id = target_tokenizer.eos_token_id
    target_tokenizer.pad_token = target_tokenizer.eos_token
    
    
    

    
    #generation kwargs setting
    generation_kwargs = {
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": agent_tokenizer.eos_token_id,
    "max_new_tokens":args.max_prompt_length,
    "min_length": -1,
    }
    
    
    #setting verbalizer ids
    verbalizer_ids=  []
    for i in range(len(verbalizer)):
        verbalizer_ids.append(agent_tokenizer.convert_tokens_to_ids(verbalizer[i]))
    
    queue = utils.TopAccuracyTextsNoDuplicates(max_size=5)
    change_num = 0
    #start training
    query_text = ''
    examples = utils.got_example_mmlu(train_dataset,verbalizer,shot=args.num_example)
    print('Inputs : ' ,examples)
    for ep in tqdm(range(args.epochs)):
        max_total_loss = 0
        min_total_loss = 0
        mean_total_loss = 0
        sum_total_loss = 0
        examples = utils.got_example_mmlu(validation_dataset,verbalizer,shot=args.num_example)
        query_text = [
            {"role" : "user", "content" : examples + 'Summarize the above input-output pairs into a single general instruction'},
        ]
            
        query_encoded = agent_tokenizer.apply_chat_template(
            query_text,
            return_tensors='pt',
            add_generation_prompt=True, #for qwen style model
            enable_thinking=False #for qwen style model
        ).view(-1).to(device)
        
        # Generate candidate prompts using the agent model (inference-only)
        input_ids = query_encoded.view(1, -1).to(device)
        response_tensors = agent_model.generate(
            input_ids,
            **generation_kwargs,
            num_return_sequences = args.prompt_per_example
        )
        # generated sequences include the input; strip input tokens to get only generated prompt
        input_len = input_ids.shape[1]
        used_prompt = [agent_tokenizer.decode(r[input_len:].squeeze(), skip_special_tokens=True) for r in response_tensors]
        
        # If many of the generated prompts are too short, exit
        if sum([len(p) for p in used_prompt]) < args.prompt_per_example * 10:
            break
        
        rewards = []
        losses = []
        accuracys,softmax_diff = utils.evaluation_sd(
            used_prompt,
            validation_dataset,
            target_model,
            target_tokenizer,
            device,
            verbalizer.values(),
            soft_diff=True,
        )
        #print(accuracys,softmax_diff)
        rewards = [  0.05 * softmax_diff[i] + 3 * accuracys[i] for i in range(len(used_prompt))]
        np_rewards = np.array(rewards)
        np_acc = np.array(accuracys)
        # convert rewards to tensors and record them
        rewards = [ torch.tensor(reward) for reward in rewards]
        for i in range(len(rewards)):
            print('reward : ', rewards[i].item(), 'acc :', accuracys[i], ' prompt : ', used_prompt[i], '\n')
            queue.add(rewards[i].item(), used_prompt[i], ep)

        # Aggregate reward stats (no RL update step; inference-only)
        rewards = torch.stack(rewards)
        mean_reward = torch.mean(rewards)
        max_reward = torch.max(rewards)
        mean_total_loss += mean_reward.item()
        max_total_loss += max_reward.item()
        min_total_loss += torch.min(rewards).item()
        sum_total_loss += torch.sum(rewards).item()

        # Log serializable metrics to wandb with explicit step
        wandb.log({
            'rewards': rewards.cpu().tolist(),
            'mean_reward': mean_reward.item(),
            'max_reward': max_reward.item(),
            'valid_acc': float(np.mean(np_acc)),
            'mean_softmax_diff': float(np.mean(np_rewards)),
        }, step=ep)
        wandb.log({
            'mean_loss' : mean_total_loss,
            'max_loss' : max_total_loss,
            'min_loss' : min_total_loss,
            'sum_loss' : sum_total_loss,
        })
                            
            
    print('Final test Start')
    prompt_queue = queue.get_top_texts()
    new_acc = utils.evaluation(
        [prompt[1] for prompt in prompt_queue],
        test_dataset,
        target_model,
        target_tokenizer,
        device,
        verbalizer.values(),
    )
    print(len(prompt_queue),new_acc)
    for i in range(len(prompt_queue)):
        print('prompt : ',prompt_queue[i][1],'acc : ',new_acc[i])
    max_new_acc = np.max(np.array(new_acc))
    wandb.log({
        'final_acc' : max_new_acc,
        'final_mean_acc' : np.mean(np.array(new_acc))
    })
    with open('results.txt',"a") as f:
        f.write(args.dataset + ' : ' + str(max_new_acc) + '\n')
            
if __name__ == '__main__':
    main()
                
                    
                    
    
    
    