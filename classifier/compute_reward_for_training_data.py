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

def load_reward_model(reward_model_dir, args):
    """
    Load the trained reward model and its tokenizer.
    """
    model_name = args.model_name
    device = args.device

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

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

    return model, tokenizer

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

def process_json_file(json_file_path, model, tokenizer, args):
    """
    Process a single JSON file and compute rewards for each output.
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    results = []

    for item in tqdm(data, desc=f"Processing {os.path.basename(json_file_path)}"):
        instruction = item['instruction']
        input_text = item.get('input', '')
        preference_prompt = item.get('preference_prompt', '')
        outputs = item['output']

        # Generate texts for all completions
        texts = [
            process_text(instruction, input_text, preference_prompt, completion)
            for completion in outputs
        ]

        # Evaluate rewards in batch
        rewards = evaluate_reward(texts, model, tokenizer, args)

        item_result = {
            'id': item['id'],
            'instruction': instruction,
            'input': input_text,
            'preference_prompt': preference_prompt,
            'outputs': outputs,
            'rewards': rewards
        }
        results.append(item_result)

    return results

def save_results(results, output_file_path):
    """
    Save the results with rewards to a JSON file.
    """
    with open(output_file_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_file_path}")

def process_dimension(dimension, args):
    """
    Process all JSON files in a given dimension directory.
    """
    print(f"Processing dimension: {dimension}")
    dimension_dir = os.path.join(args.data_dir, dimension)
    reward_model_dir = os.path.join(args.reward_model_dir, dimension, 'checkpoint-1000')
    output_dir = os.path.join(args.output_dir, dimension)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model, tokenizer = load_reward_model(reward_model_dir, args)

    json_files = [f for f in os.listdir(dimension_dir) if f.endswith('.json')]
    json_files.sort(key=lambda x: int(os.path.splitext(x)[0]))  # Sort files numerically

    for json_file in json_files:
        json_file_path = os.path.join(dimension_dir, json_file)
        output_file_path = os.path.join(output_dir, json_file)

        if os.path.exists(output_file_path):
            print(f"Skipping {json_file} as it has already been processed.")
            continue

        results = process_json_file(json_file_path, model, tokenizer, args)
        save_results(results, output_file_path)

def parse_args():
    parser = argparse.ArgumentParser(description="Process data and compute rewards.")

    # Add arguments
    parser.add_argument('--data_dir', type=str, default='classifier/data/processed/raw_generation',
                        help='Directory containing the data')
    parser.add_argument('--reward_model_dir', type=str, default='model/reward_models', help='Directory containing reward models')
    parser.add_argument('--output_dir', type=str, default='classifier/data/processed/rewarded', help='Directory to store results with rewards')
    parser.add_argument('--torch_dtype', type=str, default='float16', help='Torch data type')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run the model on')
    parser.add_argument('--model_name', type=str, default="TheBloke/tulu-7B-fp16", help='Model name')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum length for tokenization')
    parser.add_argument('--cache_dir', type=str, default='.cache', help='Directory to cache models')
    parser.add_argument('--dimensions', type=str, default='P1A', help='Dimensions to process, separated by "+"')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()  # Parse the command-line arguments
    set_seed(args.seed)

    if args.torch_dtype == 'float16':
        args.torch_dtype = torch.float16

    dimensions = args.dimensions.split('+')

    for dimension in dimensions:
        process_dimension(dimension, args)
