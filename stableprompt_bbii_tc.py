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
from dataset_utils import load_all_dataset,dataset_dicts,load_qa_dataset,qa_dicts,load_generation_dataset,load_bigbench
# PEFT/TRL removed — using inference-only models
def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_model',type=str,default='google/gemma-3-1b-it')
    parser.add_argument('--agent_model',type=str,default='google/gemma-3-1b-it')
    parser.add_argument('--task',type=str,default='classification')
    parser.add_argument('--dataset',type=str,default='implicatures')
    parser.add_argument(
        '--verbalizer',
        type = str,
        nargs = '+',
        default = None
    )
    parser.add_argument('--cache_dir',type=str,default='/mnt/sdb/llm/')
    parser.add_argument('--batch_size',type=int,default=4)
    parser.add_argument('--max_prompt_length',type=int,default=50)
    parser.add_argument('--train_data_per_labels',type=int,default=10)
    parser.add_argument('--num_example',type=int,default=5)
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--meta_prompt',type=str,
                        default = '''I gave a friend an instruction and five inputs. 
                        The friend read the instruction and wrote an output for every one of the inputs.
                        Here are the input-output pairs: \n''')
    parser.add_argument('--prompt_per_example',type=int,default=4)
    parser.add_argument('--update_term',type=int,default=11)
    parser.add_argument('--update_threshold',type=float,default=0.01)   
    parser.add_argument('--num_test_example',type=int,default=20)

    args = parser.parse_args()
    return args

def main():
    
    args = parser_args()
    device= 'cuda:0'
    wandb.init(project='BBH', 
               config=args,
               name = args.dataset+'_algprompt')
    
    metrics,train_dataset,test_dataset,verbalizer,task_prefix = load_bigbench(args.dataset)
    validation_dataset = train_dataset
    print('Verbalizer : ',verbalizer)
        
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
    if verbalizer is not None:
        verbalizer_ids=  []
        for i in range(len(verbalizer)):
            verbalizer_ids.append(agent_tokenizer.convert_tokens_to_ids(verbalizer[i]))
    
    queue = utils.TopAccuracyTextsNoDuplicates(max_size=5)
    change_num = 0
    #start training
    query_text = ''
    examples = utils.got_example_bbh(train_dataset,verbalizer,shot=args.num_example,metrics=metrics)
    print('Inputs : ' ,examples)
    pp = ['',examples,task_prefix]
    
    '''
    if metrics == 'multiple_choice_grade':
        new_acc = utils.evaluation(
            pp,
            test_dataset,
            target_model,
            target_tokenizer,
            device,
            verbalizer.values(),
        )
    else:
        new_acc,_ = utils.evaluation_generation(
            pp,
            test_dataset,
            target_model,
            target_tokenizer,
            device,
            show=True
        )
    no_prompt = new_acc[0]
    few_shot  = new_acc[1]
    prfix = new_acc[2]
    print('no prompt acc : ',no_prompt)
    print('few shot acc : ',few_shot)
    print('prefix acc : ',prfix)
    wandb.log({
        'no_prompt_acc' : no_prompt,
        'few_shot_acc' : few_shot,
        'prefix_acc' : prfix,
    })
    '''
    
    
    
    
    
    
    
    
    

    for ep in tqdm(range(args.epochs)):
        max_total_loss = 0
        min_total_loss = 0
        mean_total_loss = 0
        sum_total_loss = 0
        examples = utils.got_example_bbh(validation_dataset,verbalizer,shot=args.num_example,metrics=metrics)
        query_text = [
            {"role" : "user", "content" : args.meta_prompt + '\n' + examples},
            {"role": "assistant","content" : "The Instruction is : "}
        ]
            
        query_encoded = agent_tokenizer.apply_chat_template(
            query_text,
            return_tensors='pt'
        ).view(-1).to(device)
        
        response_tensors = agent_model.generate(
            query_encoded,
            **generation_kwargs,
            num_return_sequences = args.prompt_per_example
        )
            
        used_prompt = [agent_tokenizer.decode(r.squeeze(),skip_special_tokens=True) for r in response_tensors]
        
        # If many of the generated prompts are too short, exit
        if sum([len(p) for p in used_prompt]) < args.prompt_per_example * 10:
            break
        
        rewards = []
        losses = []
        if metrics == 'multiple_choice_grade':
            accuracys,softmax_diff = utils.evaluation_sd(
                used_prompt,
                validation_dataset,
                target_model,
                target_tokenizer,
                device,
                verbalizer.values(),
                soft_diff=True,
            )
            rewards = [  0.05 * softmax_diff[i] + 3 * accuracys[i] for i in range(len(used_prompt))]
        elif metrics == 'exact_str_match':
            rewards,accuracys = utils.evaluation_generation(
                used_prompt,
                validation_dataset,
                target_model,
                target_tokenizer,
                device,
            )
            rewars = [ rewards[i] + accuracys[i] * 1000 for i in range(len(used_prompt))] #?
            #accuracys = rewards
        #print(accuracys,softmax_diff)
        
        np_rewards = np.array(rewards)
        np_acc = np.array(accuracys)
        rewards = [ torch.tensor(reward) for reward in rewards]
        for i in range(len(rewards)):
            print('reward : ', rewards[i].item(),'acc :', accuracys[i],' prompt : ', used_prompt[i], '\n')
            queue.add(rewards[i].item(),used_prompt[i],ep)
        bs = len(np_rewards)
        #print([query_encoded.view(-1) for i in range(bs)],response_tensors,[torch.tensor(reward) for reward in rewards])
        # RL training step removed (ppo_trainer.step)
        rewards = torch.stack(rewards)
        mean_reward = torch.mean(rewards)
        max_reward = torch.max(rewards)
        mean_total_loss += mean_reward #? 平均reward
        max_total_loss += max_reward
        min_total_loss += torch.min(rewards)
        sum_total_loss += torch.sum(rewards)
        wandb.log({
            'rewards' : rewards,
            'mean_reward' : mean_reward,
            'max_reward' : max_reward,
        })
            
            
        # reference-model update removed (RL-specific)
        wandb.log({
            'mean_loss' : mean_total_loss,
            'max_loss' : max_total_loss,
            'min_loss' : min_total_loss,
            'sum_loss' : sum_total_loss,
        })
                            
            
    print('Final test Start')
    prompt_queue = queue.get_top_texts()
    if metrics == 'multiple_choice_grade':
        new_acc = utils.evaluation(
            [prompt[1] for prompt in prompt_queue],
            test_dataset,
            target_model,
            target_tokenizer,
            device,
            verbalizer.values(),
        )
    else:
        new_acc,_ = utils.evaluation_generation(
            [prompt[1] for prompt in prompt_queue],
            test_dataset,
            target_model,
            target_tokenizer,
            device,
            show=True
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
                
                    
                    
    
    
    