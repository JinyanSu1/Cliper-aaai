import os
import json
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import random
import glob
PREFERENCE_PROMPT_DICT = {
        'P1A': 'Generate a response that can be easily understood by an elementary school student.',
        'P1B': 'Generate a response that only a PhD Student in that specific field could understand.',
        'P2A': 'Generate a response that is concise and to the point without being verbose.',
        'P2B': 'Generate a response that is very informative without missing any background information.',
        'P3A': 'Generate a response that is friendly, witty, funny, and humorous, like a close friend.',
        'P3B': 'Generate a response in an unfriendly manner.'
    }
def set_seed(seed):
    """
    Set the seed for reproducibility across NumPy, PyTorch, and Python's random module.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def unify_pad_eos_ids(model, tokenizer):
    """
    Ensure model and tokenizer have the same pad and eos token IDs.
    """
    assert model.config.eos_token_id is not None, "Model must have an eos token id"
    assert model.config.eos_token_id == tokenizer.eos_token_id, "Model and tokenizer must have the same eos token id"
    if model.config.pad_token_id != tokenizer.pad_token_id:
        print('Adjusting model pad token ID to match tokenizer pad token ID.')
        assert tokenizer.pad_token_id is not None, "Tokenizer must have a pad token id"
        model.config.pad_token_id = tokenizer.pad_token_id

def freeze_trainable_parameters(model):
    """
    Freeze all trainable parameters in the model.
    """
    for param in model.parameters():
        param.requires_grad = False
    print('All model parameters have been frozen.')

def load_reward_model(reward_model_dir, tokenizer, args):
    """
    Load the trained reward model.
    """
    model_name = args.model_name
    device = args.device

    # Load the reward model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        torch_dtype=args.torch_dtype,
        load_in_4bit=True,
        cache_dir=args.cache_dir
    )

    # Adjust special tokens
    unify_pad_eos_ids(model, tokenizer)

    # Load the LoRA parameters
    model = PeftModel.from_pretrained(model, reward_model_dir)
    freeze_trainable_parameters(model)
    model.eval()
    model.to(device)

    return model

def process_text(instruction, input_text, preference_prompt, completion):
    """
    Combine instruction, input, preference prompt, and completion into a single text.
    """
    if input_text:
        text = f"<|user|>\n{instruction} {input_text} {preference_prompt}\n<|assistant|>\n{completion}"
    else:
        text = f"<|user|>\n{instruction} {preference_prompt}\n<|assistant|>\n{completion}"
    return text

def evaluate_reward(texts, model, tokenizer, args):
    """
    Evaluate the reward for a list of texts.
    """
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_length,
        padding=True
    )
    inputs = {key: value.to(args.device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        rewards = outputs.logits.squeeze(-1).tolist()
    return rewards

def process_dimension_data(dimension, eval_dim, model, tokenizer, args):
    """
    Process all JSON files in a given dimension directory and compute rewards.
    Returns a DataFrame with ids and computed rewards.
    """
    print(f"Processing dimension: {dimension}")
    dimension_dir = os.path.join(args.data_dir, dimension)

    json_files = [f for f in os.listdir(dimension_dir) if f.endswith('.json')]
    json_files.sort(key=lambda x: int(os.path.splitext(x)[0]))  # Sort files numerically

    data_rows = []

    for json_file in json_files:
        json_file_path = os.path.join(dimension_dir, json_file)

        with open(json_file_path, 'r') as f:
            data = json.load(f)

        for item in data:
            instruction = item['instruction']
            input_text = item.get('input', '')
            preference_prompt = PREFERENCE_PROMPT_DICT[eval_dim]
            outputs = item['outputs']
            id_ = item['id']

            # Assume there's only one output per entry
            completion = outputs[0]

            # Generate text
            text = process_text(instruction, input_text, preference_prompt, completion)

            # Evaluate reward
            reward = evaluate_reward([text], model, tokenizer, args)[0]

            data_rows.append({
                'id': id_,
                f'reward_{dimension}': reward
            })

    # Create DataFrame for this dimension
    df = pd.DataFrame(data_rows)
    return df

def combine_dimension_data(dimension_dfs):
    """
    Combine DataFrames from all dimensions into a single DataFrame.
    """
    # Merge DataFrames on 'id'
    from functools import reduce
    df_combined = reduce(lambda left, right: pd.merge(left, right, on='id'), dimension_dfs)
    return df_combined

def parse_args():
    parser = argparse.ArgumentParser(description="Process data and compute rewards.")

    # Add arguments
    parser.add_argument('--data_dir', type=str, default='/share/nikola/js3673/project/personalized_alignment/Classifier-Guided/data/processed_data_top1',
                        help='Directory containing the data')
    parser.add_argument('--reward_model_dir', type=str, default = '/share/nikola/js3673/project/personalized_alignment/Classifier-Guided/data_process/reward_models/', help='Directory containing the reward model')
    parser.add_argument('--output_file_dir', type=str, default='/share/nikola/js3673/project/personalized_alignment/Classifier-Guided/data/rewards', help='File to store the final matrix')
    parser.add_argument('--torch_dtype', type=str, default='float16', help='Torch data type')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run the model on')
    parser.add_argument('--model_name', type=str, default="TheBloke/tulu-7B-fp16", help='Model name')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum length for tokenization')
    parser.add_argument('--cache_dir', type=str, default='/share/nikola/js3673/cache', help='Directory to cache models')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--eval_dim', type=str, default='P1A', help='Evaluation dimension')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()  # Parse the command-line arguments
    set_seed(args.seed)

    if args.torch_dtype == 'float16':
        args.torch_dtype = torch.float16

    dimensions = ['P1A', 'P1B', 'P2A', 'P2B', 'P3A', 'P3B']

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)

    # Load reward model
    reward_model_dir = os.path.join(args.reward_model_dir, f'{args.eval_dim}/checkpoint-1000')
    model = load_reward_model(reward_model_dir, tokenizer, args)

    # Process data for each dimension and collect DataFrames
    dimension_dfs = []
    for dimension in dimensions:
        df = process_dimension_data(dimension, args.eval_dim, model, tokenizer, args)
        dimension_dfs.append(df)

    # Combine all dimension DataFrames into a single DataFrame
    df_combined = combine_dimension_data(dimension_dfs)

    # Reorder columns
    column_order = ['id'] + [f'reward_{dim}' for dim in dimensions]
    df_combined = df_combined[column_order]

    # Save the final matrix
    output_file = os.path.join(args.output_file_dir, f'rewards_{args.eval_dim}.npy')

    # Or save as NumPy array
    rewards_matrix = df_combined.drop('id', axis=1).to_numpy()
    np.save(output_file, rewards_matrix)
    print(f"Final rewards matrix saved to {output_file}")
