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
from ii_utils import TASK_TO_METRIC, evaluation_ii_batch
from dataset_utils import load_bigbench


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_model',
                        type=str,
                        default='google/gemma-1.1-7b-it')
    parser.add_argument('--agent_model',
                        type=str,
                        default='google/gemma-1.1-7b-it')
    parser.add_argument('--task', type=str, default='Generation')
    parser.add_argument('--dataset',
                        type=str,
                        default='gender_inclusive_sentences_german')
    parser.add_argument('--verbalizer', type=str, nargs='+', default=None)
    parser.add_argument('--cache_dir', type=str, default='llm')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_prompt_length', type=int, default=100)
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
                        default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()
    return args


@torch.inference_mode()
def main():

    args = parser_args()
    device = 'cuda:0'

    # set seeds for reproducibility
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
    run_name = f"{short_agent}_TO_{short_target}_bs{args.batch_size}_top{args.topk}_s{args.seed}"

    wandb.init(
        project="GFB_BBH_TG",
        group=args.dataset,
        name=run_name,
        tags=[
            args.dataset, f"agent:{short_agent}", f"target:{short_target}",
            args.task
        ],
        config=vars(args),
    )

    wandb.define_metric("epoch_summary/epoch")
    wandb.define_metric("epoch_summary/*", step_metric="epoch")
    wandb.define_metric("running_max/*", step_metric="step")
    wandb.define_metric("running_topk/*", step_metric="step")

    # load dataset
    metrics, train_dataset, test_dataset, verbalizer, task_prefix = load_bigbench(
        args.dataset)
    print('[task] :', args.dataset)
    print('[metric] :', TASK_TO_METRIC.get(args.dataset, 'em'))
    print('[train_data_size] :', len(train_dataset))
    print('[test_data_size] :', len(test_dataset))

    wandb.config.update({
        'train_data_size':
        len(train_dataset),
        'test_data_size':
        len(test_dataset),
        'num_labels':
        len(verbalizer) if verbalizer else 0,
        'used_verbalizer':
        list(verbalizer.values()) if verbalizer else 'N/A'
    })
    try:
        sample_examples = utils.got_example_bbh(train_dataset,
                                                verbalizer,
                                                shot=args.num_example,
                                                metrics=metrics)
        full_metaprompt = args.meta_prompt + '\n' + sample_examples + '\nThe Instruction is : '
        print('\n=== Example meta-prompt (used for generation) ===\n')
        print(full_metaprompt)
        print('\n=== End meta-prompt ===\n')
        if wandb.run is not None:
            wandb.run.summary['example_metaprompt'] = full_metaprompt
    except Exception as e:
        print('Warning: could not construct example metaprompt:', e)

    # make dataloader
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

    # load agent model (inference-only; RL removed)
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
        "max_new_tokens": args.max_prompt_length,
        "min_length": -1,
    }

    queue = utils.TopAccuracyTextsNoDuplicates(max_size=args.topk)
    global_step = 0
    running_max_accuracy = float('-inf')
    # start training
    print('[Training Start]')
    for ep in tqdm(range(args.epochs)):
        batch_count = 0

        for batch in train_dataloader:
            inputs = batch['text']
            labels = batch['label']
            examples = utils.got_example_bbh(train_dataset,
                                             verbalizer,
                                             shot=args.num_example,
                                             metrics=metrics)
            query_text = [{
                "role": "user",
                "content": args.meta_prompt + '\n' + examples
            }, {
                "role": "assistant",
                "content": "The Instruction is : "
            }]

            query_encoded = agent_tokenizer.apply_chat_template(
                query_text, return_tensors='pt').to(device)
            input_length = query_encoded.shape[1]

            # set deterministic seed for this generation step
            torch.manual_seed(args.seed + ep * 100000 + batch_count)
            response_tensors = agent_model.generate(
                query_encoded,
                **generation_kwargs,
                num_return_sequences=args.prompt_per_example)

            used_prompt = [
                agent_tokenizer.decode(r[input_length:],
                                       skip_special_tokens=True).strip()
                for r in response_tensors
            ]

            accuracy = []
            new_dict = {'text': inputs, 'label': labels}
            new_ds = Dataset.from_dict(new_dict)
            for prompt in used_prompt:
                acc = evaluation_ii_batch(
                    prompt,
                    new_ds,
                    target_model,
                    target_tokenizer,
                    device,
                    generation_kwargs,
                    args.dataset,
                    batch_size=16,
                )
                accuracy.append(acc)
            np_accuracy = np.array(accuracy)
            accuracy = [torch.tensor(a) for a in accuracy]
            for i in range(len(accuracy)):
                print('[accuracy] : ', accuracy[i].item(), ' [prompt] : ',
                      used_prompt[i], '\n')
                queue.add(accuracy[i].item(), used_prompt[i], ep)
            bs = len(np_accuracy)
            accuracy = torch.stack(accuracy)
            mean_accuracy_val = float(torch.mean(accuracy).item())
            max_accuracy_val = float(torch.max(accuracy).item())
            min_accuracy_val = float(
                torch.min(accuracy).item()) if bs > 0 else 0.0
            std_accuracy_val = float(
                torch.std(accuracy).item()) if bs > 1 else 0.0

            # update running max
            if max_accuracy_val > running_max_accuracy:
                running_max_accuracy = max_accuracy_val

            # compute running topk stats
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
                'batch/accuracy_mean': mean_accuracy_val,
                'batch/accuracy_max': max_accuracy_val,
                'batch/accuracy_min': min_accuracy_val,
                'batch/accuracy_std': std_accuracy_val,
                'running_max/accuracy': running_max_accuracy,
                'running_topk/avg_accuracy': running_topk_avg,
                'running_topk/min_accuracy': running_topk_min,
                'global_step': global_step,
                'step': global_step,
                'epoch': ep,
            }
            wandb.log(metrics, step=global_step)
            global_step += 1
            batch_count += 1

    print('[Final test Start]')
    prompt_queue = queue.get_top_texts()
    used_prompt = [prompt[1] for prompt in prompt_queue]
    new_scores = []
    for prompt in used_prompt:
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
        new_scores.append(float(score))
    print(len(prompt_queue))
    print(len(new_scores))

    for i in range(len(prompt_queue)):
        print('prompt : ', prompt_queue[i][1], 'score : ', new_scores[i])
    if new_scores:
        max_new_score = float(np.max(np.array(new_scores)))
        mean_new_score = float(np.mean(np.array(new_scores)))
    else:
        max_new_score = 0.0
        mean_new_score = 0.0

    final_results_table = wandb.Table(
        columns=["Rank", "Prompt", "Score", "Accuracy", "Epoch"],
        data=[[
            i + 1, prompt_queue[i][1], new_scores[i], prompt_queue[i][0],
            prompt_queue[i][2] if len(prompt_queue[i]) > 2 else "N/A"
        ] for i in range(len(prompt_queue))])

    wandb.log({
        'final/best_score': max_new_score,
        'final/mean_score': mean_new_score,
        'final/results_table': final_results_table,
    })

    try:
        best_idx = int(np.argmax(np.array(new_scores)))
        wandb.run.summary["best_prompt"] = prompt_queue[best_idx][1]
        wandb.run.summary["best_score"] = max_new_score
        wandb.run.summary["mean_score"] = mean_new_score
    except Exception:
        pass

    wandb.finish()


if __name__ == '__main__':
    main()
