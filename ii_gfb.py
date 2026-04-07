import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import numpy as np
import wandb
import random
import utils
from datasets import Dataset
from ii_utils import load_ii_data, got_example_ii, TASK_TO_METRIC, evaluation_ii_batch


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_model',
                        type=str,
                        default='google/gemma-1.1-7b-it')
    parser.add_argument('--agent_model',
                        type=str,
                        default='google/gemma-1.1-7b-it')
    parser.add_argument('--task', type=str, default='ii')
    parser.add_argument('--dataset', type=str, default='active_to_passive')
    parser.add_argument('--cache_dir', type=str, default='llm')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_prompt_length', type=int, default=50)
    parser.add_argument('--train_data_per_labels', type=int, default=16)
    parser.add_argument('--num_example', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument(
        '--meta_prompt',
        type=str,
        default='''I gave a friend an instruction and five inputs.
                        The friend read the instruction and wrote an output for every one of the inputs.
                        Here are the input-output pairs: \n
                        ''',
    )
    parser.add_argument('--prompt_per_example', type=int, default=4)
    parser.add_argument('--topk',
                        type=int,
                        default=5,
                        help='Max size of top prompts to keep')
    parser.add_argument('--seed',
                        type=int,
                        default=6,
                        help='Random seed for reproducibility')

    return parser.parse_args()


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


@torch.inference_mode()
def main():
    args = parser_args()
    device = 'cuda:0'

    seed_everything(args.seed)

    # 1. 高可读性模型名、运行名生成（来自 tc_gfb）
    def get_short_name(model_path):
        return model_path.split('/')[-1]

    short_agent = get_short_name(args.agent_model)
    short_target = get_short_name(args.target_model)
    run_name = f"{short_agent}_TO_{short_target}_bs{args.batch_size}_top{args.topk}_s{args.seed}"

    # 2. WandB 初始化（tc_gfb 统一规范）
    wandb.init(
        project=f"GFB_{args.task.upper()}",
        group=args.dataset,
        name=run_name,
        tags=[
            args.dataset,
            f"agent:{short_agent}",
            f"target:{short_target}",
            args.task,
        ],
        config=vars(args),
    )

    wandb.define_metric('epoch_summary/epoch')
    wandb.define_metric('epoch_summary/*', step_metric='epoch')
    wandb.define_metric('running_max/*', step_metric='step')
    wandb.define_metric('running_topk/*', step_metric='step')

    # 3. 数据准备
    train_dataset, test_dataset, validation_dataset = load_ii_data(
        args.dataset)

    print('[task] :', args.dataset)
    print('[metric] :', TASK_TO_METRIC.get(args.dataset, 'em'))
    print('[train_data_size] :', len(train_dataset))
    print('[test_data_size] :', len(test_dataset))
    print('[validation_data_size] :', len(validation_dataset))

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True)

    # Log dataset info
    wandb.config.update({
        'train_data_size': len(train_dataset),
        'test_data_size': len(test_dataset),
        'validation_data_size': len(validation_dataset),
        'metric': TASK_TO_METRIC.get(args.dataset, 'em')
    })

    # 4. 记录示例 meta_prompt
    try:
        sample_examples = got_example_ii(validation_dataset,
                                         shot=args.num_example)
        full_metaprompt = args.meta_prompt + '\n' + sample_examples + '\nThe Instruction is : '
        print('\n=== Example meta-prompt (used for generation) ===\n')
        print(full_metaprompt)
        print('\n=== End meta-prompt ===\n')
        if wandb.run is not None:
            wandb.run.summary['example_metaprompt'] = full_metaprompt
    except Exception as e:
        print('Warning: could not construct example metaprompt:', e)

    # 5. 模型加载（同 tc_gfb）
    agent_tokenizer = AutoTokenizer.from_pretrained(args.agent_model,
                                                    cache_dir=args.cache_dir)
    agent_model = AutoModelForCausalLM.from_pretrained(
        args.agent_model,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        cache_dir=args.cache_dir,
    )
    agent_tokenizer.pad_token = agent_tokenizer.eos_token

    target_tokenizer = AutoTokenizer.from_pretrained(args.target_model,
                                                     cache_dir=args.cache_dir)
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16,
        device_map='auto',
    )
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
    running_max_accuracy = float('-inf')

    # 6. 生成 + 评估循环
    for ep in tqdm(range(args.epochs)):
        epoch_accuracies = []

        for batch in train_dataloader:
            inputs = batch['text']
            labels = batch['label']

            examples = got_example_ii(validation_dataset,
                                      shot=args.num_example)
            query_text = [
                {
                    'role': 'user',
                    'content': args.meta_prompt + '\n' + examples,
                },
                {
                    'role': 'assistant',
                    'content': 'The Instruction is : ',
                },
            ]

            query_encoded = agent_tokenizer.apply_chat_template(
                query_text, return_tensors='pt').to(device)
            input_length = query_encoded.shape[1]

            torch.manual_seed(args.seed + ep * 100000 + global_step)
            response_tensors = agent_model.generate(
                query_encoded,
                **generation_kwargs,
                num_return_sequences=args.prompt_per_example,
            )

            used_prompt = [
                agent_tokenizer.decode(r[input_length:],
                                       skip_special_tokens=True).strip()
                for r in response_tensors
            ]

            torch.cuda.empty_cache()

            accuracies = []
            new_dict = {'text': inputs, 'label': labels}
            new_ds = Dataset.from_dict(new_dict)
            for prompt in used_prompt:
                score = evaluation_ii_batch(
                    prompt,
                    new_ds,
                    target_model,
                    target_tokenizer,
                    device,
                    generation_kwargs,
                    args.dataset,
                    batch_size=16,
                )
                accuracies.append(float(score))

            accuracies_tensor = torch.tensor(accuracies)
            epoch_accuracies.extend(accuracies)

            for i in range(len(used_prompt)):
                print('[accuracy] :', accuracies[i], '[prompt] :',
                      used_prompt[i])
                queue.add(accuracies[i], used_prompt[i], ep)

            max_acc = float(torch.max(accuracies_tensor))
            running_max_accuracy = max(running_max_accuracy, max_acc)

            topk_texts = queue.get_top_texts()
            if topk_texts:
                topk_scores = np.array([item[0] for item in topk_texts],
                                       dtype=float)
                running_topk_avg = float(np.mean(topk_scores))
                running_topk_min = float(np.min(topk_scores))
            else:
                running_topk_avg = 0.0
                running_topk_min = 0.0

            wandb.log(
                {
                    'batch/accuracy_mean': float(
                        torch.mean(accuracies_tensor)),
                    'batch/accuracy_max': float(torch.max(accuracies_tensor)),
                    'batch/accuracy_min': float(torch.min(accuracies_tensor)),
                    'batch/accuracy_std': float(torch.std(accuracies_tensor)),
                    'running_max/accuracy': running_max_accuracy,
                    'running_topk/avg_accuracy': running_topk_avg,
                    'running_topk/min_accuracy': running_topk_min,
                    'global_step': global_step,
                    'step': global_step,
                    'epoch': ep,
                },
                step=global_step,
            )
            global_step += 1

        if epoch_accuracies:
            epoch_acc_np = np.array(epoch_accuracies)
            wandb.log({
                'epoch':
                int(ep),
                'epoch_summary/mean_accuracy':
                float(np.mean(epoch_acc_np)),
                'epoch_summary/max_accuracy':
                float(np.max(epoch_acc_np)),
                'epoch_summary/min_accuracy':
                float(np.min(epoch_acc_np)),
                'epoch_summary/std_accuracy':
                float(np.std(epoch_acc_np)),
                'epoch_summary/num_batches':
                len(train_dataloader),
            })

    # 7. 最终测试（相当 tc_gfb 的 final evaluation）
    print('[Final test Start]')
    prompt_queue = queue.get_top_texts()
    final_prompts = [item[1] for item in prompt_queue]

    new_acc = []
    for prompt in final_prompts:
        score = evaluation_ii_batch(
            prompt,
            test_dataset,
            target_model,
            target_tokenizer,
            device,
            generation_kwargs,
            args.dataset,
            batch_size=16,
        )
        new_acc.append(float(score))

    for i, (acc_value, text, ep) in enumerate(prompt_queue):
        print('[prompt] :', text, '[accuracy] :', new_acc[i], '[epoch] :', ep)

    max_new_acc = float(np.max(np.array(new_acc))) if new_acc else 0.0
    mean_new_acc = float(np.mean(np.array(new_acc))) if new_acc else 0.0

    final_results_table = wandb.Table(
        columns=['Rank', 'Prompt', 'Accuracy', 'Epoch'],
        data=[[
            i + 1, final_prompts[i], new_acc[i],
            prompt_queue[i][2] if len(prompt_queue[i]) > 2 else 'N/A'
        ] for i in range(len(final_prompts))],
    )

    wandb.log({
        'final/best_accuracy':
        max_new_acc,
        'final/mean_accuracy':
        mean_new_acc,
        'final/accuracy_std':
        float(np.std(np.array(new_acc))) if new_acc else 0.0,
        'final/results_table':
        final_results_table,
    })

    if new_acc:
        best_idx = int(np.argmax(np.array(new_acc)))
        wandb.run.summary['best_prompt'] = final_prompts[best_idx]
        wandb.run.summary['best_accuracy'] = max_new_acc
        wandb.run.summary['mean_accuracy'] = mean_new_acc

    wandb.finish()


if __name__ == '__main__':
    main()
