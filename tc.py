# HyperParameters Stableprompt  
# Learning Rate 1.00E-05 
# Value loss Coefficient 0.1 
# Gamma 1 
# GAE Lambda 0.95 
# cliprange 0.2 
# ut 5  
# Update Threshold(%) 0.05 
# Rollback Threshold(%) 0.1 
# Prompt per Batch 4 
# Maximum Prompt Length 150 
# ca 10 
# cs 0.1  
# Table 6: Detail parameters used in StablePrompt.
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer,AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig,AutoModelForCausalLMWithValueHead
import argparse
import numpy as np
import wandb
import copy
import random
import heapq
import utils
from dataset_utils import load_all_dataset,dataset_dicts,load_qa_dataset,qa_dicts,load_generation_dataset
from peft import LoraConfig
from datasets import Dataset

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_model',type=str,default='google/gemma-1.1-2b-it')
    parser.add_argument('--agent_model',type=str,default='google/gemma-1.1-2b-it')
    parser.add_argument('--task',type=str,default='classification')
    parser.add_argument('--dataset',type=str,default='sst2')
    parser.add_argument(
        '--verbalizer',
        type = str,
        nargs = '+',
        default = None
    )
    parser.add_argument('--cache_dir',type=str,default='llm')
    parser.add_argument('--batch_size',type=int,default=16)
    parser.add_argument('--max_prompt_length',type=int,default=150)  # changed to 150
    parser.add_argument('--train_data_per_labels',type=int,default=16)
    parser.add_argument('--num_example',type=int,default=5)
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--meta_prompt',type=str,
                        default = '''I gave a friend an instruction and five inputs. 
                        The friend read the instruction and wrote an output for every one of the inputs.
                        Here are the input-output pairs: \n
                        ''',)
    parser.add_argument('--prompt_per_example',type=int,default=4)
    parser.add_argument('--update_term',type=int,default=5)   # ut = 5
    parser.add_argument('--update_threshold',type=float,default=0.05)   
    parser.add_argument('--num_test_example',type=int,default=20)

    # new hyperparams from the table
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--value_loss_coef', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--clip_range', type=float, default=0.2)
    parser.add_argument('--rollback_threshold', type=float, default=0.1)
    parser.add_argument('--ca', type=float, default=10.0)
    parser.add_argument('--cs', type=float, default=0.1)

    args = parser.parse_args()
    return args

def main():
    
    args = parser_args()
    device= 'cuda:0'
    
    # Enhanced wandb initialization with config tracking
    wandb.init(
        project='algprompt_' + args.task + '_' + args.dataset, 
        config={
            'task': args.task,
            'dataset': args.dataset,
            'target_model': args.target_model,
            'agent_model': args.agent_model,
            'learning_rate': args.learning_rate,
            'value_loss_coef': args.value_loss_coef,
            'gamma': args.gamma,
            'gae_lambda': args.gae_lambda,
            'clip_range': args.clip_range,
            'update_term': args.update_term,
            'update_threshold': args.update_threshold,
            'rollback_threshold': args.rollback_threshold,
            'prompt_per_batch': args.prompt_per_example,
            'max_prompt_length': args.max_prompt_length,
            'ca': args.ca,
            'cs': args.cs,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'num_example': args.num_example,
            'train_data_per_labels': args.train_data_per_labels,
        },
        name=args.task + '_' + args.dataset + '_' + args.agent_model + '_' + args.target_model
    )
    # Define custom x-axis for epoch-level metrics
    wandb.define_metric("epoch_summary/epoch")
    wandb.define_metric("epoch_summary/*", step_metric="epoch_summary/epoch")
    wandb.define_metric("model_update/epoch")
    wandb.define_metric("model_update/*", step_metric="model_update/epoch")
    
    # Log hyperparameters as a table
    hyperparam_table = wandb.Table(
        columns=["Parameter", "Value"],
        data=[
            ["Learning Rate", args.learning_rate],
            ["Value Loss Coef", args.value_loss_coef],
            ["Gamma", args.gamma],
            ["GAE Lambda", args.gae_lambda],
            ["Clip Range", args.clip_range],
            ["Update Term (ut)", args.update_term],
            ["Update Threshold", args.update_threshold],
            ["Rollback Threshold", args.rollback_threshold],
            ["Prompt per Batch", args.prompt_per_example],
            ["Max Prompt Length", args.max_prompt_length],
            ["ca", args.ca],
            ["cs", args.cs],
        ]
    )
    wandb.log({"hyperparameters": hyperparam_table})
    
    if args.task == 'classification':
        dataset = load_all_dataset(args.dataset)
        train_dataset = dataset[0]
        test_dataset = dataset[2]
        #test_dataset = utils.create_balanced_subset(test_dataset,100)
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
        test_dataset = utils.create_balanced_subset(test_dataset,100)
        if args.verbalizer is None:
            verbalizer = qa_dicts()
        num_labels = len(verbalizer)
        validation_dataset = train_dataset
    
    elif args.task == 'generation':
        dataset = load_generation_dataset(args.dataset)
        train_dataset = dataset[0]
        test_dataset = dataset[2]
        test_dataset = utils.create_balanced_subset(test_dataset,100)
        verbalizer = None
        validation_dataset = train_dataset
        
    #make dataloader
    test_dataloader = DataLoader(test_dataset,batch_size = args.batch_size,shuffle = True)
    train_dataloader = DataLoader(train_dataset,batch_size = args.batch_size,shuffle = True)
    
    print('train_data_size' , len(train_dataset))
    print('test_data_size' , len(test_dataset))
    
    # Log dataset info
    wandb.log({
        'train_data_size': len(train_dataset),
        'test_data_size': len(test_dataset),
        'num_labels': len(verbalizer) if verbalizer else 0,
    })
    
        #load agent model
    config = PPOConfig(
        model_name = args.agent_model,
        learning_rate = args.learning_rate,            # use table lr
        batch_size = args.prompt_per_example,
        mini_batch_size= args.prompt_per_example,
        log_with='wandb',
        vf_coef = args.value_loss_coef,
        gamma = args.gamma,
        lam = args.gae_lambda,
        cliprange = args.clip_range,
    )

    lora_config = LoraConfig(
        r= 16,
        lora_alpha = 32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    agent_tokenizer = AutoTokenizer.from_pretrained(args.agent_model,cache_dir = args.cache_dir)
    agent_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.agent_model,
        torch_dtype=torch.bfloat16,
        device_map = 'auto',
        peft_config = lora_config,
        cache_dir = args.cache_dir
    )
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.agent_model,
        torch_dtype=torch.bfloat16,
        device_map = 'auto',
        peft_config = lora_config,
        cache_dir = args.cache_dir
    )
    agent_tokenizer.pad_token = agent_tokenizer.eos_token
    ppo_trainer = PPOTrainer(config,agent_model,ref_model,agent_tokenizer)
    
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
    "max_new_tokens": args.max_prompt_length,   # uses updated default 150
    "min_length": -1,
    }
    
    
    #setting verbalizer ids
    verbalizer_ids=  []
    for i in range(len(verbalizer)):
        verbalizer_ids.append(agent_tokenizer.convert_tokens_to_ids(verbalizer[i]))
    
    queue = utils.TopAccuracyTextsNoDuplicates(max_size=5)
    change_num = 0
    global_step = 0
    
    #start training
    for ep in tqdm(range(args.epochs)):
        max_total_loss = 0
        min_total_loss = 0
        mean_total_loss = 0
        sum_total_loss = 0
        
        epoch_rewards = []
        epoch_accuracies = []
        epoch_prompt_lengths = []
        batch_count = 0
        
        for batch in train_dataloader:
            inputs = batch['text']
            labels = batch['label']
            examples = utils.got_example(validation_dataset,verbalizer,shot=args.num_example)
            with torch.no_grad():
                
                query_text = [
                    {"role" : "user", "content" : args.meta_prompt + '\n' + examples},
                    {"role": "assistant","content" : "The Instruction is : "}
                ]
                
                query_encoded = agent_tokenizer.apply_chat_template(
                    query_text,
                    return_tensors='pt'
                ).view(-1).to(device)
                
                response_tensors =ppo_trainer.generate(
                    query_encoded,
                    **generation_kwargs,
                    return_prompt=False,
                    num_return_sequences = args.prompt_per_example
                )
                
                used_prompt = [agent_tokenizer.decode(r.squeeze(),skip_special_tokens=True) for r in response_tensors]
                
            # If many of the generated prompts are too short, exit
            if sum([len(p) for p in used_prompt]) < args.prompt_per_example * 10:
                break
            
            rewards = []
            losses = []
            new_dict ={
                'text' : inputs,
                'label' : labels
            }
            new_ds = Dataset.from_dict(new_dict)
            with torch.no_grad(): 
                accuracys,softmax_diff = utils.evaluation_sd(
                    used_prompt,
                    new_ds,
                    target_model,
                    target_tokenizer,
                    'cuda:0',
                    verbalizer.values(),
                )
            rewards = [  args.cs * softmax_diff[i] + args.ca * accuracys[i] for i in range(len(used_prompt))]
            np_rewards = np.array(rewards)
            np_acc = np.array(accuracys)
            rewards = [ torch.tensor(reward) for reward in rewards]
            
            # Track prompt lengths
            prompt_lengths = [len(p) for p in used_prompt]
            epoch_prompt_lengths.extend(prompt_lengths)
            
            for i in range(len(rewards)):
                print('reward : ', rewards[i].item(),'acc :', accuracys[i],' prompt : ', used_prompt[i], '\n')
                queue.add(rewards[i].item(),used_prompt[i],ep)
            bs = len(np_rewards)
            
            stats = ppo_trainer.step([query_encoded.view(-1) for i in range(bs)],
                         [response for response in response_tensors],
                         rewards)
            
            rewards = torch.stack(rewards)
            mean_reward = torch.mean(rewards)
            max_reward = torch.max(rewards)
            min_reward = torch.min(rewards)
            std_reward = torch.std(rewards)
            
            # Collect epoch stats
            epoch_rewards.extend(rewards.tolist())
            epoch_accuracies.extend(accuracys)
            batch_count += 1
            global_step += 1
            
            # Detailed step-level logging
            log_dict = {
                'step': global_step,
                'epoch': ep,
                'batch': batch_count,
                # Reward metrics
                'reward/mean': mean_reward.item(),
                'reward/max': max_reward.item(),
                'reward/min': min_reward.item(),
                'reward/std': std_reward.item(),
                # Accuracy metrics
                'accuracy/mean': np.mean(np_acc),
                'accuracy/max': np.max(np_acc),
                'accuracy/min': np.min(np_acc),
                'accuracy/std': np.std(np_acc),
                # Softmax diff metrics
                'softmax_diff/mean': np.mean(softmax_diff),
                'softmax_diff/max': np.max(softmax_diff),
                # Prompt metrics
                'prompt/mean_length': np.mean(prompt_lengths),
                'prompt/max_length': np.max(prompt_lengths),
                'prompt/min_length': np.min(prompt_lengths),
            }
            
            # Log PPO stats if available
            if stats:
                for key, value in stats.items():
                    if isinstance(value, (int, float)):
                        log_dict[f'PPO/{key}'] = value
                    elif isinstance(value, torch.Tensor):
                        log_dict[f'PPO/{key}'] = value.mean().item()
            
            wandb.log(log_dict, step=global_step)
        
        # Epoch-level summary
        if epoch_rewards:
            epoch_rewards_np = np.array(epoch_rewards)
            epoch_acc_np = np.array(epoch_accuracies)
            
            wandb.log({
                'epoch_summary/epoch': ep,
                'epoch_summary/mean_reward': np.mean(epoch_rewards_np),
                'epoch_summary/max_reward': np.max(epoch_rewards_np),
                'epoch_summary/min_reward': np.min(epoch_rewards_np),
                'epoch_summary/std_reward': np.std(epoch_rewards_np),
                'epoch_summary/mean_accuracy': np.mean(epoch_acc_np),
                'epoch_summary/max_accuracy': np.max(epoch_acc_np),
                'epoch_summary/num_batches': batch_count,
                'epoch_summary/mean_prompt_length': np.mean(epoch_prompt_lengths),
                # Histograms
                'epoch_summary/reward_histogram': wandb.Histogram(epoch_rewards_np),
                'epoch_summary/accuracy_histogram': wandb.Histogram(epoch_acc_np),
                'epoch_summary/prompt_length_histogram': wandb.Histogram(epoch_prompt_lengths),
            })
            
        #reference model update
        if ep % args.update_term == 0 and ep!=0:
            response_tensors,ref_response_tensors = ppo_trainer.generate(query_encoded.view(-1),**generation_kwargs,return_prompt=False, num_return_sequences=2,generate_ref_response=True)
            used_prompt = [agent_tokenizer.decode(r.squeeze(),skip_special_tokens=True) for r in response_tensors]
            ref_used_prompt = [agent_tokenizer.decode(r.squeeze(),skip_special_tokens=True) for r in ref_response_tensors]
            acc = utils.evaluation(
                used_prompt,
                validation_dataset,
                target_model,
                target_tokenizer,
                device,
                verbalizer.values(),
            )
            ref_acc = utils.evaluation(
                ref_used_prompt,
                validation_dataset,
                target_model,
                target_tokenizer,
                device,
                verbalizer.values(),
            )
            print('acc : ', acc)
            print('ref_acc : ', ref_acc)
            mean_acc = np.mean(np.array(acc))
            mean_ref_acc = np.mean(np.array(ref_acc))
            diff = mean_acc - mean_ref_acc
            
            update_action = "none"
            if diff > args.update_threshold:
                ppo_trainer.ref_model =  ppo_trainer.model
                print('update ref model')
                change_num +=1
                update_action = "update"
            elif diff < -args.rollback_threshold:
                ppo_trainer.model = ppo_trainer.ref_model
                print('rollback model')
                change_num -=1
                update_action = "rollback"
            
            
            wandb.log({
                'model_update/epoch': ep,
                'model_update/change_num': change_num,
                'model_update/valid_acc': mean_acc,
                'model_update/ref_valid_acc': mean_ref_acc,
                'model_update/acc_diff': diff,
                'model_update/action': wandb.Table(
                    columns=["epoch", "action", "diff"],
                    data=[[ep, update_action, diff]]
                ),
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
    
    # Create final results table
    final_results_table = wandb.Table(
        columns=["Rank", "Prompt", "Accuracy", "Reward", "Epoch"],
        data=[
            [i+1, prompt_queue[i][1], new_acc[i], prompt_queue[i][0], prompt_queue[i][2] if len(prompt_queue[i]) > 2 else "N/A"]
            for i in range(len(prompt_queue))
        ]
    )
    
    for i in range(len(prompt_queue)):
        print('prompt : ',prompt_queue[i][1],'acc : ',new_acc[i])
    max_new_acc = np.max(np.array(new_acc))
    mean_new_acc = np.mean(np.array(new_acc))
    
    # Final summary logging
    wandb.log({
        'final/best_accuracy': max_new_acc,
        'final/mean_accuracy': mean_new_acc,
        'final/accuracy_std': np.std(np.array(new_acc)),
        'final/results_table': final_results_table,
        'final/total_model_updates': change_num,
    })
    
    # Log best prompt as summary
    best_idx = np.argmax(np.array(new_acc))
    wandb.run.summary["best_prompt"] = prompt_queue[best_idx][1]
    wandb.run.summary["best_accuracy"] = max_new_acc
    wandb.run.summary["mean_accuracy"] = mean_new_acc
    
    wandb.finish()
    
            
if __name__ == '__main__':
    main()

