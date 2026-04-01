import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import numpy as np
import wandb
import random
import utils
from dataset_utils import load_bigbench
# removed reinforcement-learning dependencies (trl, peft) to match ii_gfb flow


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_model',
                        type=str,
                        default='google/gemma-1.1-7b-it')
    parser.add_argument('--agent_model',
                        type=str,
                        default='google/gemma-1.1-7b-it')
    parser.add_argument('--task', type=str, default='Multiple Choice')
    parser.add_argument('--dataset', type=str, default='implicatures')
    parser.add_argument('--verbalizer', type=str, nargs='+', default=None)
    parser.add_argument('--cache_dir', type=str, default='llm')
    parser.add_argument('--max_prompt_length', type=int, default=100)
    parser.add_argument('--train_data_per_labels', type=int, default=10)
    parser.add_argument('--num_example', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument(
        '--meta_prompt',
        type=str,
        default='''I gave a friend an instruction and five inputs. 
                        The friend read the instruction and wrote an output for every one of the inputs.
                        Here are the input-output pairs: \n''')
    parser.add_argument('--prompt_per_example', type=int, default=4)
    parser.add_argument('--topk',
                        type=int,
                        default=5,
                        help='Max size of top prompts to keep')
    parser.add_argument('--seed',
                        type=int,
                        default=6,
                        help='Random seed for reproducibility')

    args = parser.parse_args()
    return args


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

    def get_short_name(model_path):
        return model_path.split('/')[-1]

    short_agent = get_short_name(args.agent_model)
    short_target = get_short_name(args.target_model)
    run_name = f"{short_agent}_TO_{short_target}_top{args.topk}_s{args.seed}"

    wandb.init(
        project='GFB_BBH_TC',
        group=args.dataset,
        name=run_name,
        tags=[
            args.dataset, f"agent:{short_agent}", f"target:{short_target}",
            args.task
        ],
        config=vars(args),
    )
    wandb.define_metric('epoch_summary/epoch')
    wandb.define_metric('epoch_summary/*', step_metric='epoch')
    wandb.define_metric('running_max/*', step_metric='global_step')
    wandb.define_metric('running_topk/*', step_metric='global_step')

    metrics, train_dataset, test_dataset, verbalizer, task_prefix = load_bigbench(
        args.dataset)

    validation_dataset = train_dataset

    print('[task] :', args.dataset)
    print('[Verbalizer] : ', verbalizer)
    print('[Train Dataset Size] : ', len(train_dataset))
    print('[Test Dataset Size] : ', len(test_dataset))
    print('[Validation Dataset Size] : ', len(validation_dataset))

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
        sample_examples = utils.got_example_bbh(validation_dataset,
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

    # load agent model (non-RL flow)
    agent_tokenizer = AutoTokenizer.from_pretrained(args.agent_model,
                                                    cache_dir=args.cache_dir)
    agent_model = AutoModelForCausalLM.from_pretrained(
        args.agent_model,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        cache_dir=args.cache_dir,
    )
    agent_tokenizer.pad_token = agent_tokenizer.eos_token
    agent_model.eval()

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

    # setting verbalizer ids
    if verbalizer is not None:
        verbalizer_ids = []
        for i in range(len(verbalizer)):
            verbalizer_ids.append(
                agent_tokenizer.convert_tokens_to_ids(verbalizer[i]))

    queue = utils.TopAccuracyTextsNoDuplicates(max_size=args.topk)
    global_step = 0
    running_max_reward = float('-inf')

    # start training
    print('[Training Start]')
    for ep in tqdm(range(args.epochs)):
        epoch_accuracies = []
        examples = utils.got_example_bbh(validation_dataset,
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
            query_text, return_tensors='pt')
        # handle tokenizer return types and move tensors to device
        if isinstance(query_encoded, dict):
            input_ids = query_encoded.get('input_ids').to(device)
        else:
            input_ids = query_encoded.to(device)
        input_length = input_ids.shape[1]

        torch.manual_seed(args.seed + ep * 100000 + global_step)
        response_tensors = agent_model.generate(
            input_ids,
            **generation_kwargs,
            num_return_sequences=args.prompt_per_example,
        )

        used_prompt = [
            agent_tokenizer.decode(r[input_length:],
                                   skip_special_tokens=True).strip()
            for r in response_tensors
        ]

        # evaluate generated prompts (no RL)
        if metrics == 'multiple_choice_grade':
            accuracys_tensors, _ = utils.evaluation_sd(
                used_prompt,
                validation_dataset,
                target_model,
                target_tokenizer,
                device,
                verbalizer.values(),
                soft_diff=True,
            )
            accuracys = [float(x.item()) for x in accuracys_tensors]
        else:
            _, accuracys = utils.evaluation_generation(
                used_prompt,
                validation_dataset,
                target_model,
                target_tokenizer,
                device,
            )
            accuracys = [float(x) for x in accuracys]

        for i in range(len(accuracys)):
            print('[Score] : ', accuracys[i], ' [Prompt] : ', used_prompt[i])
            queue.add(accuracys[i], used_prompt[i], ep)

        accuracies_tensor = torch.tensor(accuracys)
        epoch_accuracies.extend(accuracys)

        max_acc = float(
            torch.max(accuracies_tensor)) if len(accuracys) > 0 else 0.0
        running_max_reward = max(running_max_reward, max_acc)

        # running/topk summary
        topk_texts = queue.get_top_texts()
        if topk_texts:
            topk_scores = np.array([item[0] for item in topk_texts],
                                   dtype=float)
            running_topk_avg = float(np.mean(topk_scores))
            running_topk_min = float(np.min(topk_scores))
        else:
            running_topk_avg = 0.0
            running_topk_min = 0.0

        # epoch summary
        epoch_acc_np = np.array(epoch_accuracies)
        wandb.log(
            {
                'epoch': ep,
                'epoch_summary/mean_accuracy': float(np.mean(epoch_acc_np)),
                'epoch_summary/max_accuracy': float(np.max(epoch_acc_np)),
                'epoch_summary/min_accuracy': float(np.min(epoch_acc_np)),
                'epoch_summary/std_accuracy': float(np.std(epoch_acc_np)),
                'running_topk/avg_accuracy': running_topk_avg,
                'running_topk/min_accuracy': running_topk_min,
                'running_max/accuracy': running_max_reward,
                'global_step': global_step,
            },
            step=global_step)
        global_step += 1

    print('[Final test Start]')
    prompt_queue = queue.get_top_texts()
    final_prompts = [item[1] for item in prompt_queue]

    if final_prompts:
        if metrics == 'multiple_choice_grade':
            new_acc = utils.evaluation(
                final_prompts,
                test_dataset,
                target_model,
                target_tokenizer,
                device,
                verbalizer.values(),
            )
        else:
            new_acc, _ = utils.evaluation_generation(final_prompts,
                                                     test_dataset,
                                                     target_model,
                                                     target_tokenizer,
                                                     device,
                                                     show=True)

        for i, (score, text, ep) in enumerate(prompt_queue):
            print('[prompt] :', text, '[final_score] :', new_acc[i].item(),
                  '[score] :', score, '[epoch] :', ep)

        max_new_acc = float(np.max(np.array(new_acc))) if new_acc else 0.0
        mean_new_acc = float(np.mean(np.array(new_acc))) if new_acc else 0.0

        final_results_table = wandb.Table(
            columns=[
                'Rank', 'Prompt', 'Final Score', 'Original Score', 'Epoch'
            ],
            data=[
                [
                    i + 1,
                    final_prompts[i],
                    new_acc[i],
                    prompt_queue[i][0],  # Original score
                    prompt_queue[i][2] if len(prompt_queue[i]) > 2 else 'N/A'
                ] for i in range(len(final_prompts))
            ],
        )

        wandb.log({
            'final/best_score':
            max_new_acc,
            'final/mean_score':
            mean_new_acc,
            'final/score_std':
            float(np.std(np.array(new_acc))) if new_acc else 0.0,
            'final/results_table':
            final_results_table,
        })

        if new_acc:
            best_idx = int(np.argmax(np.array(new_acc)))
            if wandb.run is not None:
                wandb.run.summary['best_prompt'] = final_prompts[best_idx]
                wandb.run.summary['best_score'] = max_new_acc
                wandb.run.summary['mean_score'] = mean_new_acc

    wandb.finish()


if __name__ == '__main__':
    main()
