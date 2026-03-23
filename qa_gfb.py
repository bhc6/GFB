import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import numpy as np
import wandb
import random
import utils
from dataset_utils import load_qa_dataset, qa_dicts


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_model',
                        type=str,
                        default='google/gemma-1.1-7b-it')
    parser.add_argument('--agent_model',
                        type=str,
                        default='google/gemma-1.1-7b-it')
    parser.add_argument('--task', type=str, default='qa')
    parser.add_argument('--dataset', type=str, default='squad')
    parser.add_argument('--verbalizer', type=str, nargs='+', default=None)
    parser.add_argument('--cache_dir', type=str, default='llm')
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--max_prompt_length', type=int, default=150)
    parser.add_argument('--num_example', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument(
        '--meta_prompt',
        type=str,
        default='''I gave a friend an instruction and five inputs. 
                        The friend read the instruction and wrote an output for every one of the inputs.
                        Here are the input-output pairs: \n''')
    parser.add_argument('--prompt_per_example', type=int, default=4)
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--ca', type=float, default=10.0)
    parser.add_argument('--cs', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    return args


@torch.inference_mode()
def main():
    args = parser_args()
    device = 'cuda:0'

    def seed_everything(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        try:
            torch.cuda.manual_seed_all(seed)
        except Exception:
            pass
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass

    seed_everything(args.seed)

    def get_short_name(model_path):
        return model_path.split('/')[-1]

    short_agent = get_short_name(args.agent_model)
    short_target = get_short_name(args.target_model)
    run_name = f"{short_agent}_TO_{short_target}_s{args.seed}_bs{args.batch_size}_top{args.topk}"
    wandb.init(project=f"GFB_{args.task.upper()}",
               group=args.dataset,
               name=run_name,
               tags=[
                   args.dataset, f"agent:{short_agent}",
                   f"target:{short_target}", args.task
               ],
               config=vars(args))
    wandb.define_metric("epoch_summary/epoch")
    wandb.define_metric("epoch_summary/*", step_metric="epoch")
    wandb.define_metric("running_max/*", step_metric="step")
    wandb.define_metric("running_topk/*", step_metric="step")

    if args.task == 'qa':
        dataset = load_qa_dataset(args.dataset)
        train_dataset = dataset[0]
        test_dataset = dataset[2]
        validation_dataset = dataset[4]
        if args.verbalizer is None:
            verbalizer = qa_dicts()
        else:
            verbalizer = args.verbalizer
    else:
        raise ValueError(f"Task '{args.task}' is not supported.")

    def worker_init_fn(worker_id):
        worker_seed = args.seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        try:
            torch.manual_seed(worker_seed)
        except Exception:
            pass

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  worker_init_fn=worker_init_fn)

    print('[train_data_size]', len(train_dataset))
    print('[test_data_size]', len(test_dataset))
    print('[validation_data_size]', len(validation_dataset))

    wandb.config.update({
        'train_data_size':
        len(train_dataset),
        'test_data_size':
        len(test_dataset),
        'validation_data_size':
        len(validation_dataset),
        'num_labels':
        len(verbalizer) if verbalizer else 0,
        'used_verbalizer':
        list(verbalizer.values()) if verbalizer else 'N/A'
    })
    try:
        sample_examples = utils.got_example(validation_dataset,
                                            verbalizer,
                                            shot=args.num_example,
                                            seed=args.seed)
        full_metaprompt = args.meta_prompt + '\n' + sample_examples + '\nThe Instruction is : '
        print('\n=== Example meta-prompt (used for generation) ===\n')
        print(full_metaprompt)
        print('\n=== End meta-prompt ===\n')
        if wandb.run is not None:
            wandb.run.summary['example_metaprompt'] = full_metaprompt
    except Exception as e:
        print('Warning: could not construct example metaprompt:', e)

    agent_tokenizer = AutoTokenizer.from_pretrained(args.agent_model,
                                                    cache_dir=args.cache_dir)
    agent_model = AutoModelForCausalLM.from_pretrained(
        args.agent_model,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        cache_dir=args.cache_dir)
    agent_tokenizer.pad_token = agent_tokenizer.eos_token

    target_tokenizer = AutoTokenizer.from_pretrained(args.target_model,
                                                     cache_dir=args.cache_dir)
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16,
        device_map='auto')
    target_model.config.pad_token_id = target_tokenizer.eos_token_id
    target_tokenizer.pad_token = target_tokenizer.eos_token

    generation_kwargs = {
        'top_k': 0.0,
        'top_p': 1.0,
        'do_sample': True,
        'pad_token_id': agent_tokenizer.eos_token_id,
        'max_new_tokens': args.max_prompt_length,
        'min_length': -1,
    }

    queue = utils.TopAccuracyTextsNoDuplicates(max_size=args.topk)
    global_step = 0
    running_max_reward = float('-inf')
    running_max_accuracy = float('-inf')

    for ep in tqdm(range(args.epochs)):
        epoch_rewards = []
        epoch_accuracies = []
        epoch_prompt_lengths = []
        epoch_softmax_diff = []
        batch_count = 0
        for batch in train_dataloader:
            inputs = batch['text']
            labels = batch['label']
            examples = utils.got_example(validation_dataset,
                                         verbalizer,
                                         shot=args.num_example,
                                         seed=args.seed + ep * 100000 +
                                         batch_count)
            query_text = [{
                "role": "user",
                "content": args.meta_prompt + '\n' + examples
            }, {
                "role": "assistant",
                "content": "The Instruction is : "
            }]
            query_encoded = agent_tokenizer.apply_chat_template(
                query_text, return_tensors='pt').to(device)
            torch.manual_seed(args.seed + ep * 100000 + batch_count)
            response_tensors = agent_model.generate(
                query_encoded,
                **generation_kwargs,
                num_return_sequences=args.prompt_per_example)
            input_length = query_encoded.shape[1]
            used_prompt = [
                agent_tokenizer.decode(r[input_length:],
                                       skip_special_tokens=True).strip()
                for r in response_tensors
            ]

            torch.cuda.empty_cache()

            softmax_diff, accuracys = utils.evaluation_soft(
                prompts=used_prompt,
                inputs=inputs,
                targets=labels,
                model=target_model,
                tokenizer=target_tokenizer,
                device=device,
                verbalizer=list(verbalizer.values()),
                side='First',
                return_reward=False)

            rewards = [
                args.cs * softmax_diff[i] + args.ca * accuracys[i]
                for i in range(len(used_prompt))
            ]

            np_acc = np.array(accuracys)
            prompt_lengths = [len(p) for p in used_prompt]
            epoch_prompt_lengths.extend(prompt_lengths)

            for i in range(len(rewards)):
                try:
                    r_val = rewards[i].item() if hasattr(
                        rewards[i], 'item') else float(rewards[i])
                except Exception:
                    r_val = float(rewards[i])
                print('[reward] : ', r_val, '[accuracy] :', accuracys[i],
                      '[softmax_diff] : ', softmax_diff[i].item(),
                      '[prompt] : ', used_prompt[i], '\n')
                queue.add(r_val, used_prompt[i], ep)

            rewards_tensor = torch.stack([torch.as_tensor(r) for r in rewards])
            max_reward = torch.max(rewards_tensor)
            epoch_rewards.extend(rewards_tensor.tolist())
            epoch_accuracies.extend(accuracys)
            epoch_softmax_diff.extend([
                x.float().item() if torch.is_tensor(x) else float(x)
                for x in softmax_diff
            ])
            if max_reward.item() > running_max_reward:
                running_max_reward = max_reward.item()
            if np.max(np_acc) > running_max_accuracy:
                running_max_accuracy = np.max(np_acc)

            # Compute topk running average reward from the global queue
            topk_texts = queue.get_top_texts()
            if topk_texts:
                topk_rewards = np.array([item[0] for item in topk_texts],
                                        dtype=float)
                running_topk_avg = float(np.mean(topk_rewards))
                running_topk_min = float(np.min(topk_rewards))
            else:
                running_topk_avg = 0.0
                running_topk_min = 0.0

            metrics = {
                'batch/reward_mean': float(torch.mean(rewards_tensor)),
                'batch/reward_max': float(torch.max(rewards_tensor)),
                'batch/reward_min': float(torch.min(rewards_tensor)),
                'batch/reward_std': float(torch.std(rewards_tensor)),
                'batch/accuracy_mean': float(np.mean(epoch_accuracies)),
                'batch/accuracy_max': float(np.max(epoch_accuracies)),
                'batch/accuracy_min': float(np.min(epoch_accuracies)),
                'batch/softmax_diff_mean': float(np.mean(epoch_softmax_diff)),
                'batch/prompt_length_mean':
                float(np.mean(epoch_prompt_lengths)),
                'running_max/reward': running_max_reward,
                'running_max/accuracy': running_max_accuracy,
                'running_topk/avg_reward': running_topk_avg,
                'running_topk/min_reward': running_topk_min,
                'global_step': global_step,
                'step': global_step,
                'epoch': ep,
            }
            wandb.log(metrics, step=global_step)
            global_step += 1
            batch_count += 1
        if epoch_rewards:
            epoch_rewards_np = np.array(epoch_rewards)
            epoch_acc_np = np.array(epoch_accuracies)
            epoch_softmax_diff_np = np.array(epoch_softmax_diff)
            wandb.log({
                'epoch':
                int(ep),
                'epoch_summary/mean_reward':
                float(np.mean(epoch_rewards_np)),
                'epoch_summary/max_reward':
                float(np.max(epoch_rewards_np)),
                'epoch_summary/min_reward':
                float(np.min(epoch_rewards_np)),
                'epoch_summary/std_reward':
                float(np.std(epoch_rewards_np)),
                'epoch_summary/mean_accuracy':
                float(np.mean(epoch_acc_np)),
                'epoch_summary/max_accuracy':
                float(np.max(epoch_acc_np)),
                'epoch_summary/mean_softmax_diff':
                float(np.mean(epoch_softmax_diff_np)),
                'epoch_summary/num_batches':
                int(batch_count),
                'epoch_summary/mean_prompt_length':
                float(np.mean(epoch_prompt_lengths)),
            })
    print('[Final test Start]')
    prompt_queue = queue.get_top_texts()
    new_acc = utils.evaluation([prompt[1] for prompt in prompt_queue],
                               test_dataset, target_model, target_tokenizer,
                               device, list(verbalizer.values()))
    final_results_table = wandb.Table(
        columns=["Rank", "Prompt", "Accuracy", "Reward", "Epoch"],
        data=[[
            i + 1, prompt_queue[i][1], new_acc[i].item() if hasattr(
                new_acc[i], 'item') else float(new_acc[i]), prompt_queue[i][0],
            prompt_queue[i][2] if len(prompt_queue[i]) > 2 else "N/A"
        ] for i in range(len(prompt_queue))])
    for i in range(len(prompt_queue)):
        print(
            '[prompt] : ', prompt_queue[i][1], '[accuracy] : ',
            new_acc[i].item() if hasattr(new_acc[i], 'item') else float(
                new_acc[i]), '[reward] : ', prompt_queue[i][0], '[epoch] : ',
            prompt_queue[i][2] if len(prompt_queue[i]) > 2 else "N/A", '\n')
    try:
        new_acc_arr = np.array(new_acc, dtype=float)
    except Exception:
        new_acc_arr = np.array([float(x) for x in new_acc], dtype=float)
    wandb.log({
        'final/best_accuracy': float(np.max(new_acc_arr)),
        'final/mean_accuracy': float(np.mean(new_acc_arr)),
        'final/accuracy_std': float(np.std(new_acc_arr)),
        'final/results_table': final_results_table
    })
    best_idx = int(np.argmax(np.array(new_acc_arr)))
    wandb.run.summary['best_prompt'] = prompt_queue[best_idx][1]
    wandb.run.summary['best_accuracy'] = float(np.max(new_acc_arr))
    wandb.run.summary['mean_accuracy'] = float(np.mean(new_acc_arr))
    wandb.finish()


if __name__ == '__main__':
    main()
