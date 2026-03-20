import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import numpy as np
import wandb
import atexit
import random
import utils
from dataset_utils import load_all_dataset, dataset_dicts, load_qa_dataset, qa_dicts, load_generation_dataset
from datasets import Dataset


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_model',
                        type=str,
                        default='google/gemma-1.1-2b-it')
    parser.add_argument('--agent_model',
                        type=str,
                        default='google/gemma-1.1-2b-it')
    parser.add_argument('--task', type=str, default='tc')
    parser.add_argument('--dataset', type=str, default='sst2')
    parser.add_argument('--verbalizer', type=str, nargs='+', default=None)
    parser.add_argument('--cache_dir', type=str, default='llm')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_prompt_length', type=int,
                        default=150)  # changed to 150
    parser.add_argument('--train_data_per_labels', type=int, default=16)
    parser.add_argument('--num_example', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument(
        '--meta_prompt',
        type=str,
        default='''I gave a friend an instruction and five inputs. 
                        The friend read the instruction and wrote an output for every one of the inputs.
                        Here are the input-output pairs: \n
                        ''',
    )
    parser.add_argument('--prompt_per_example', type=int, default=4)
    parser.add_argument('--num_test_example', type=int, default=20)
    parser.add_argument('--topk',
                        type=int,
                        default=5,
                        help='Max size of top prompts to keep')
    parser.add_argument('--ca',
                        type=float,
                        default=10,
                        help='Coefficient for accuracy in reward calculation')
    parser.add_argument(
        '--cs',
        type=float,
        default=0.1,
        help='Coefficient for softmax difference in reward calculation')
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()
    return args


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

    # Apply global seed for reproducibility
    seed_everything(args.seed)

    # Enhanced wandb initialization with config tracking
    wandb.init(
        project=args.dataset + '_' + args.task + '_GFB',
        config=vars(args),
        name=args.dataset + '_' + args.agent_model + '/' + args.target_model,
    )
    # Define custom x-axis for epoch-level metrics
    wandb.define_metric("epoch_summary/epoch")
    wandb.define_metric("epoch_summary/*", step_metric="epoch_summary/epoch")
    # Define running_max metrics so they can be plotted by step or epoch
    wandb.define_metric("running_max/*", step_metric="step")
    wandb.define_metric("running_max/*", step_metric="epoch")
    # Ensure wandb finishes cleanly on program exit (including exceptions)
    atexit.register(wandb.finish)

    if args.task == 'tc':
        dataset = load_all_dataset(args.dataset)
        train_dataset = dataset[0]
        test_dataset = dataset[2]
        # test_dataset = utils.create_balanced_subset(test_dataset,100)
        if args.verbalizer is None:
            verbalizer = dataset_dicts(args.dataset)
        num_labels = len(verbalizer)
        train_dataset, validation_dataset = utils.create_balanced_subset_and_validation(
            train_dataset,
            args.train_data_per_labels * num_labels,
        )
    elif args.task == 'qa':
        dataset = load_qa_dataset(args.dataset)
        train_dataset = dataset[0]
        test_dataset = dataset[2]
        test_dataset = utils.create_balanced_subset(test_dataset, 100)
        if args.verbalizer is None:
            verbalizer = qa_dicts()
        num_labels = len(verbalizer)
        validation_dataset = train_dataset

    elif args.task == 'generation':
        dataset = load_generation_dataset(args.dataset)
        train_dataset = dataset[0]
        test_dataset = dataset[2]
        test_dataset = utils.create_balanced_subset(test_dataset, 100)
        verbalizer = None
        validation_dataset = train_dataset

    # make dataloader
    def worker_init_fn(worker_id):
        worker_seed = args.seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        try:
            torch.manual_seed(worker_seed)
        except Exception:
            pass

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        worker_init_fn=worker_init_fn,
    )

    print('[train_data_size]', len(train_dataset))
    print('[test_data_size]', len(test_dataset))

    # Log dataset info
    wandb.log({
        'train_data_size': len(train_dataset),
        'test_data_size': len(test_dataset),
        'num_labels': len(verbalizer) if verbalizer else 0,
    })

    # Display a complete example meta-prompt before training starts
    try:
        sample_examples = utils.got_example(validation_dataset,
                                            verbalizer,
                                            shot=args.num_example)
        full_metaprompt = args.meta_prompt + '\n' + sample_examples + '\nThe Instruction is : '
        print('\n=== Example meta-prompt (used for generation) ===\n')
        print(full_metaprompt)
        print('\n=== End meta-prompt ===\n')
        # store a copy in wandb summary for easy access in the run
        if wandb.run is not None:
            wandb.run.summary['example_metaprompt'] = full_metaprompt
    except Exception as e:
        print('Warning: could not construct example metaprompt:', e)

    # load agent model (inference-only; RL components removed)
    agent_tokenizer = AutoTokenizer.from_pretrained(args.agent_model,
                                                    cache_dir=args.cache_dir)
    agent_model = AutoModelForCausalLM.from_pretrained(
        args.agent_model,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        cache_dir=args.cache_dir)
    agent_tokenizer.pad_token = agent_tokenizer.eos_token

    # load target model
    target_tokenizer = AutoTokenizer.from_pretrained(args.target_model,
                                                     cache_dir=args.cache_dir)
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16,
        device_map='auto')
    target_model.config.pad_token_id = target_tokenizer.eos_token_id
    target_tokenizer.pad_token = target_tokenizer.eos_token

    # generation kwargs setting
    generation_kwargs = {
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": agent_tokenizer.eos_token_id,
        "max_new_tokens": args.max_prompt_length,  # uses updated default 150
        "min_length": -1,
    }

    # setting verbalizer ids
    verbalizer_ids = []
    for i in range(len(verbalizer)):
        verbalizer_ids.append(
            agent_tokenizer.convert_tokens_to_ids(verbalizer[i]))

    queue = utils.TopAccuracyTextsNoDuplicates(max_size=args.topk)
    change_num = 0
    global_step = 0
    # running max trackers (over steps)
    running_max_reward = float('-inf')
    running_max_accuracy = float('-inf')
    print('[Start sampling]')

    for ep in tqdm(range(args.epochs)):
        epoch_rewards = []
        epoch_accuracies = []
        epoch_prompt_lengths = []
        batch_count = 0

        for batch in train_dataloader:
            inputs = batch['text']
            labels = batch['label']
            examples = utils.got_example(validation_dataset,
                                         verbalizer,
                                         shot=args.num_example)
            with torch.no_grad():

                query_text = [{
                    "role": "user",
                    "content": args.meta_prompt + '\n' + examples
                }, {
                    "role": "assistant",
                    "content": "The Instruction is : "
                }]

                query_encoded = agent_tokenizer.apply_chat_template(
                    query_text, return_tensors='pt').to(device)

                # Apply the specific seed right before generation
                torch.manual_seed(args.seed + ep * 100000 + batch_count)
                
                response_tensors = agent_model.generate(
                    query_encoded,
                    **generation_kwargs,
                    num_return_sequences=args.prompt_per_example
                )

                used_prompt = [
                    agent_tokenizer.decode(r.squeeze(),
                                           skip_special_tokens=True)
                    for r in response_tensors
                ]

            rewards = []
            new_dict = {'text': inputs, 'label': labels}
            new_ds = Dataset.from_dict(new_dict)
            with torch.no_grad():
                accuracys, softmax_diff = utils.evaluation_sd(
                    used_prompt,
                    new_ds,
                    target_model,
                    target_tokenizer,
                    'cuda:0',
                    verbalizer.values(),
                )
            rewards = [
                args.cs * softmax_diff[i] + args.ca * accuracys[i]
                for i in range(len(used_prompt))
            ]
            np_acc = np.array(accuracys)
            rewards = [torch.tensor(reward) for reward in rewards]

            # Track prompt lengths
            prompt_lengths = [len(p) for p in used_prompt]
            epoch_prompt_lengths.extend(prompt_lengths)

            for i in range(len(rewards)):
                print('[reward] : ', rewards[i].item(), '[accuracy] :',
                      accuracys[i], '[prompt] : ', used_prompt[i], '\n')
                queue.add(rewards[i].item(), used_prompt[i], ep)

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

            # softmax_diff may be a list-like of numbers/tensors
            try:
                sd_arr = np.array(softmax_diff, dtype=float)
            except Exception:
                sd_arr = np.array([float(x) for x in softmax_diff],
                                  dtype=float)

            log_dict = {
                'step': int(global_step),
                'epoch': int(ep),
                # Reward metrics (already native via .item())
                'reward/mean': float(mean_reward.item()),
                'reward/max': float(max_reward.item()),
                'reward/min': float(min_reward.item()),
                'reward/std': float(std_reward.item()),
                # Accuracy metrics
                'accuracy/mean': float(np.mean(np_acc)),
                'accuracy/max': float(np.max(np_acc)),
                'accuracy/min': float(np.min(np_acc)),
                'accuracy/std': float(np.std(np_acc)),
                # Softmax diff metrics
                'softmax_diff/mean': float(np.mean(sd_arr)),
                'softmax_diff/max': float(np.max(sd_arr)),
                # Prompt metrics
                'prompt/mean_length': float(np.mean(prompt_lengths)),
                'prompt/max_length': int(np.max(prompt_lengths)),
                'prompt/min_length': int(np.min(prompt_lengths)),
            }
            # Log step-level numeric metrics (no histograms, no sample table)
            wandb.log(log_dict, step=global_step)

            # Update running_max_reward
            if max_reward.item() > running_max_reward:
                running_max_reward = max_reward.item()

            # Log running_max metrics
            wandb.log({
                "running_max/reward": running_max_reward,
                "running_max/accuracy": running_max_accuracy,
                "step": global_step
            })

        # Epoch-level summary
        if epoch_rewards:
            epoch_rewards_np = np.array(epoch_rewards)
            epoch_acc_np = np.array(epoch_accuracies)

            wandb.log({
                'epoch_summary/epoch':
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
                'epoch_summary/num_batches':
                int(batch_count),
                'epoch_summary/mean_prompt_length':
                float(np.mean(epoch_prompt_lengths)),
            })

    print('[Final test Start]')
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
        data=[[
            i + 1, prompt_queue[i][1], new_acc[i], prompt_queue[i][0],
            prompt_queue[i][2] if len(prompt_queue[i]) > 2 else "N/A"
        ] for i in range(len(prompt_queue))])

    for i in range(len(prompt_queue)):
        print('[prompt] : ', prompt_queue[i][1], '[accuracy] : ', new_acc[i],
              '[reward] : ', prompt_queue[i][0], '[epoch] : ',
              prompt_queue[i][2] if len(prompt_queue[i]) > 2 else "N/A", '\n')
    max_new_acc = np.max(np.array(new_acc))
    mean_new_acc = np.mean(np.array(new_acc))

    # Final summary logging
    try:
        new_acc_arr = np.array(new_acc, dtype=float)
    except Exception:
        new_acc_arr = np.array([float(x) for x in new_acc], dtype=float)

    wandb.log({
        'final/best_accuracy': float(np.max(new_acc_arr)),
        'final/mean_accuracy': float(np.mean(new_acc_arr)),
        'final/accuracy_std': float(np.std(new_acc_arr)),
        'final/results_table': final_results_table,
        'final/total_model_updates': int(change_num),
    })

    # Log best prompt as summary
    best_idx = np.argmax(np.array(new_acc))
    wandb.run.summary["best_prompt"] = prompt_queue[best_idx][1]
    wandb.run.summary["best_accuracy"] = max_new_acc
    wandb.run.summary["mean_accuracy"] = mean_new_acc

    wandb.finish()


if __name__ == '__main__':
    main()
