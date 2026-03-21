# 限制在validation_dataset上的prompt优化
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import numpy as np
import wandb
import random
import utils
from dataset_utils import load_all_dataset, dataset_dicts, load_qa_dataset, qa_dicts, load_generation_dataset


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_model',
                        type=str,
                        default='google/gemma-1.1-7b-it')
    parser.add_argument('--agent_model',
                        type=str,
                        default='google/gemma-1.1-7b-it')
    parser.add_argument('--task', type=str, default='qa')
    parser.add_argument('--dataset', type=str, default='sst2')
    parser.add_argument('--verbalizer', type=str, nargs='+', default=None)
    parser.add_argument('--cache_dir', type=str, default='/mnt/sdb/llm/')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_prompt_length', type=int, default=100)
    parser.add_argument('--train_data_per_labels', type=int, default=10)
    parser.add_argument('--num_example', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument(
        '--meta_prompt',
        type=str,
        default='''I gave a friend an instruction and three inputs. 
                        The friend read the instruction and wrote an output for every one of the inputs.
                        Here are the input-output pairs: \n''')
    parser.add_argument('--prompt_per_example', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)  #42

    parser.add_argument(
        '--results_suffix',
        type=str,
        default=None,
        help='Suffix for results file, e.g. a batch run name or timestamp')
    args = parser.parse_args()
    return args


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """Worker init function for DataLoader to ensure reproducibility"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main():

    args = parser_args()
    set_seed(args.seed)

    device = 'cuda:0'
    wandb.init(project='mmlu',
               config=args,
               name=args.dataset + 'meta_change' + args.agent_model + '_' +
               args.target_model)

    if args.task == 'classification':
        dataset = load_all_dataset(args.dataset)
        test_dataset = dataset[2]
        test_dataset = utils.create_balanced_subset(test_dataset, 100)
        if args.verbalizer is None:
            verbalizer = dataset_dicts(args.dataset)
        num_labels = len(verbalizer)
        _, validation_dataset = utils.create_balanced_subset_and_validation(
            dataset[0],
            args.train_data_per_labels * num_labels,
        )
    elif args.task == 'qa':
        dataset = load_qa_dataset(args.dataset)
        test_dataset = dataset[2]
        #test_dataset = utils.create_balanced_subset(test_dataset,100)
        if args.verbalizer is None:
            verbalizer = qa_dicts()
        num_labels = len(verbalizer)
        validation_dataset = dataset[4]

    elif args.task == 'generation':
        dataset = load_generation_dataset(args.dataset)
        test_dataset = dataset[2]
        test_dataset = utils.create_balanced_subset(test_dataset, 100)
        verbalizer = None
        validation_dataset = dataset[0]

    print('test dataset size : ', len(test_dataset))
    print('validation dataset size : ', len(validation_dataset))

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

    # setting verbalizer ids
    verbalizer_ids = []
    for i in range(len(verbalizer)):
        verbalizer_ids.append(
            agent_tokenizer.convert_tokens_to_ids(verbalizer[i]))

    queue = utils.TopAccuracyTextsNoDuplicates(max_size=5)
    # Print one example at the start (from validation set)
    example_for_show = utils.got_example_mmlu(validation_dataset,
                                              verbalizer,
                                              shot=args.num_example)
    query_text_example = [{
        "role": "user",
        "content": args.meta_prompt + '\n' + example_for_show
    }, {
        "role": "assistant",
        "content": "The Instruction is : "
    }]
    print('[Example query_text] :', query_text_example)

    # start sampling
    for ep in tqdm(range(args.epochs)):
        examples = utils.got_example_mmlu(validation_dataset,
                                          verbalizer,
                                          shot=args.num_example)
        query_text = [{
            "role": "user",
            "content": args.meta_prompt + '\n' + examples
        }, {
            "role": "assistant",
            "content": "The Instruction is : "
        }]

        query_encoded = agent_tokenizer.apply_chat_template(
            query_text, return_tensors='pt').to(device)

        response_tensors = agent_model.generate(
            query_encoded,
            **generation_kwargs,
            num_return_sequences=args.prompt_per_example)
        used_prompt = []
        for r in response_tensors:
            full_response = agent_tokenizer.decode(r.squeeze(),
                                                   skip_special_tokens=True)
            if "The Instruction is :" in full_response:
                instruction = full_response.split(
                    "The Instruction is :")[-1].strip()
            else:
                instruction = full_response.strip()
            used_prompt.append(instruction)

        rewards = []
        accuracys, softmax_diff = utils.evaluation_sd(
            used_prompt,
            validation_dataset,
            target_model,
            target_tokenizer,
            device,
            verbalizer.values(),
            soft_diff=True,
        )
        # print(accuracys,softmax_diff)
        rewards = [
            0.05 * softmax_diff[i] + 3 * accuracys[i]
            for i in range(len(used_prompt))
        ]
        
        rewards = [torch.tensor(reward) for reward in rewards]
        for i in range(len(rewards)):
            print('[Reward] : ', rewards[i].item(), '[Accuracy] :',
                  accuracys[i], '[Prompt] : ', used_prompt[i], '\n')
            queue.add(rewards[i].item(), used_prompt[i], ep)
        # print([query_encoded.view(-1) for i in range(bs)],response_tensors,[torch.tensor(reward) for reward in rewards])
        rewards = torch.stack(rewards)
        mean_reward = torch.mean(rewards)
        max_reward = torch.max(rewards)
        wandb.log({
            'rewards': rewards,
            'mean_reward': mean_reward,
            'max_reward': max_reward,
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
    print(len(prompt_queue), new_acc)

    # Create wandb table for final results
    final_results_table = wandb.Table(columns=["Prompt", "Accuracy", "Epoch"])

    for i in range(len(prompt_queue)):
        print('prompt : ', prompt_queue[i][1], 'acc : ', new_acc[i])
        # prompt_queue[i] is (reward, text, ep)
        final_results_table.add_data(
            prompt_queue[i][1],  # prompt text
            new_acc[i],  # accuracy
            prompt_queue[i][2]  # epoch
        )

    max_new_acc = np.max(np.array(new_acc))
    wandb.log({
        'final_acc': max_new_acc,
        'final_mean_acc': np.mean(np.array(new_acc)),
        'final_results': final_results_table
    })

    # Determine results file name
    if args.results_suffix:
        results_file = f"results_{args.results_suffix}.txt"
    else:
        results_file = "results.txt"
    with open(results_file, "a") as f:
        f.write(args.dataset + ' : ' + str(max_new_acc) + '\n')


if __name__ == '__main__':
    main()
