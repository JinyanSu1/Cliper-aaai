import os
import numpy as np
import pandas as pd 
import json
import argparse
from argparse import Namespace
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from tqdm import tqdm
import random
from openai import OpenAI
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
    # we change model to have the same pad and eos ids as tokenizer
    assert model.config.eos_token_id is not None, "Model must have an eos token id"
    assert model.config.eos_token_id == tokenizer.eos_token_id, "Model and tokenizer must have the same eos token id"
    if model.config.pad_token_id != tokenizer.pad_token_id:
        print('we need them to be the same. For now change model pad id to the tokenizer pad id')
        assert tokenizer.pad_token_id is not None, "Tokenizer must have a pad token id"
        model.config.pad_token_id = tokenizer.pad_token_id
def freeze_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            # print(name)
            trainable_params += param.numel()
            param.requires_grad = False
    print(
        f"previous trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}. Now froze all params.")
    print('froze all params')
def load_reward_model(reward_model_dir, args):
    """
    Load the trained reward model and its tokenizer.
    """
    model_name = args.model_name
    device = args.device
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    # Load the reward model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1, torch_dtype = args.torch_dtype, load_in_4bit =True, cache_dir=args.cache_dir)
    
    # Apply any required special token adjustments (as done during training)
    unify_pad_eos_ids(model, tokenizer)
    model = PeftModel.from_pretrained(model, reward_model_dir)
    # Load the LoRA configuration if applicable (as done in training)
    # If you applied LoRA during training, ensure you load the same config.
    freeze_trainable_parameters(model)
    model.eval()
    model.to(device)
    return model, tokenizer
def process_text(prompt, completion_a, completion_b):
    text_a = f"<|user|>\n{prompt}\n<|assistant|>\n{completion_a}"
    text_b = f"<|user|>\n{prompt}\n<|assistant|>\n{completion_b}"
    return [text_a, text_b]
def evaluate_reward(texts, model, tokenizer, args):
    rewards = []
    for _ in range(args.n_evals):
        inputs = tokenizer(texts, return_tensors="pt", truncation=True, max_length=args.max_length, padding=True)
        inputs = {key: value.to(args.device) for key, value in inputs.items()}
        reward = model(**inputs)[0].squeeze().tolist()
        rewards.append(reward)
    rewards = torch.tensor(rewards)
    mean_reward = rewards.mean().item()
    std_reward = rewards.std().item()
    return mean_reward, std_reward
def evaluate_reward_df(df, model, tokenizer, args, dimension):
    mean_a = []
    std_a = []
    mean_b = []
    std_b = []
    rewarded_raw_data_dir = args.rewarded_raw_data_dir
    output_file_path = os.path.join(rewarded_raw_data_dir, f'{dimension}.json')
    if os.path.exists(output_file_path):
        existing_df = pd.read_json(output_file_path, orient='records', lines=True)
        processed_rows = len(existing_df)
        print(f"Resuming from {processed_rows} processed rows")
    else:
        processed_rows = 0
        existing_df = pd.DataFrame()
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="generating rewards"):
        prompt = row['user_input']
        completion_a = row['completion_a']
        completion_b = row['completion_b']
        texts = process_text(prompt, completion_a, completion_b)
        mean_reward_a, std_reward_a = evaluate_reward(texts[0], model, tokenizer, args)
        mean_a.append(mean_reward_a)
        std_a.append(std_reward_a)
        
        mean_reward_b, std_reward_b = evaluate_reward(texts[1], model, tokenizer, args)
        mean_b.append(mean_reward_b)
        std_b.append(std_reward_b)
        if (idx + 1) % 10 == 0:  # Save after every 10 rows, adjust this number as needed
            df.loc[:idx, 'mean_reward_a'] = mean_a
            df.loc[:idx, 'std_reward_a'] = std_a
            df.loc[:idx, 'mean_reward_b'] = mean_b
            df.loc[:idx, 'std_reward_b'] = std_b
            # Concatenate current results with any previous data
            df_to_save = pd.concat([existing_df, df.loc[:idx]], ignore_index=True)
            df_to_save.to_json(output_file_path, orient='records', lines=True)

            
    # Final save after completing all rows
    df.loc[:idx, 'mean_reward_a'] = mean_a
    df.loc[:idx, 'std_reward_a'] = std_a
    df.loc[:idx, 'mean_reward_b'] = mean_b
    df.loc[:idx, 'std_reward_b'] = std_b
    # Concatenate current results with any previous data
    df_to_save = pd.concat([existing_df, df.loc[:idx]], ignore_index=True)
    df_to_save.to_json(output_file_path, orient='records', lines=True)
    print(f"Final save completed for {dimension}")
    
    return df_to_save
def save_rewarded_raw_data(df, args, dimension):
    """
    Save the rewarded raw data if needed.
    """
    rewarded_raw_data_dir = args.rewarded_raw_data_dir
    if not os.path.exists(rewarded_raw_data_dir):
        os.makedirs(rewarded_raw_data_dir)

    output_file_path = os.path.join(rewarded_raw_data_dir, f'{dimension}.json')
    df.to_json(output_file_path, orient='records', lines=True)
    print(f"Saved rewarded raw data for {dimension} at {output_file_path}")

def filter_and_save_top_texts(df, args, dimension):
    personalized_prompt_map = {'P1A': ' Generate a response that can be easily understood by an elementary school student.',
                               'P1B': ' Generate a response that only a PhD Student in that specific field could understand.',
                               'P2A': ' Generate a response that is concise and to the point without being verbose.',
                               'P2B': ' Generate a response that is very informative without missing any background information.',
                               'P3A': ' Generate a response that is friendly, witty, funny, and humorous, like a close friend.',
                               'P3B': ' Generate a response in an unfriendly manner.'}
        
        
        
    
    personalized_prompt = personalized_prompt_map[dimension]
    if not os.path.exists(args.filtered_data_dir):
        os.makedirs(args.filtered_data_dir)
    df_a = pd.DataFrame({
        'prompt': df['user_input'],
        'text': df['completion_a'],
        'reward': df['mean_reward_a']

    })
    
    df_b = pd.DataFrame({
        'prompt': df['user_input'],
        'text': df['completion_b'],
        'reward': df['mean_reward_b']

    })

    # Combine the two DataFrames vertically (i.e., stacking them on top of each other)
    combined_df = pd.concat([df_a, df_b], ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=['text'])
    # Sort the DataFrame by the reward column in descending order
    combined_df_sorted = combined_df.sort_values(by='reward', ascending=False)
    combined_df_sorted['prompt'] = combined_df_sorted['prompt'].str.replace(personalized_prompt, '')
    if args.top_n > 1:
        top_n = args.top_n
    else:
        top_n = int(len(df) * args.top_n)
    if top_n > len(df):
        print(f"Warning: top_n ({top_n}) is greater than the number of texts ({len(df)}). Will use all the data")
    df_top_n = combined_df_sorted.iloc[:top_n]
    df_top_n['label'] =dimension
    df_top_n = df_top_n[['prompt', 'text', 'label']].to_dict(orient='records')
    output_file_path = os.path.join(args.filtered_data_dir, f'{dimension}.json')

    # Save the top texts as a JSON file
    with open(output_file_path, 'w') as f:
        json.dump(df_top_n, f, indent=4)
    print(f'Saved top {top_n} texts for dimension {dimension} to {output_file_path}')

    




def load_raw_data(dimension, args):
    print('loading raw data:', dimension)
    data_path = os.path.join(args.raw_data_dir, f'{dimension}.json')
    with open(data_path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)



def process_dimension(dimension, args):
    """
    Process the data for a single dimension: load the model, load the raw data, filter by reward, and save the result.
    """
    
    model_dir = os.path.join(args.reward_model_dir, dimension, 'checkpoint-1000')
    # Load reward model and raw data
    model, tokenizer = load_reward_model(model_dir, args)
    if not os.path.exists(args.rewarded_raw_data_dir):
        os.makedirs(args.rewarded_raw_data_dir)
    rewarded_data_path = os.path.join(args.rewarded_raw_data_dir, f'{dimension}.json')
    # check if the rewarded raw data exists
    if os.path.exists(rewarded_data_path):
        rewarded_df = pd.read_json(rewarded_data_path, orient='records', lines=True)
        
    else:
        rewarded_df = pd.DataFrame()
    if not args.total_number_of_texts:
        args.total_number_of_texts = len(load_raw_data(dimension, args))
        print(f"get the reward for {dimension}: total number of texts{args.total_number_of_texts}")
    if len(rewarded_df) < args.total_number_of_texts:
        df_raw = load_raw_data(dimension, args)
        df_raw = df_raw.iloc[len(rewarded_df):args.total_number_of_texts]
        rewarded_df = evaluate_reward_df(df_raw, model, tokenizer, args, dimension)
        
        
    rewarded_df = rewarded_df.iloc[:args.total_number_of_texts]
    filter_and_save_top_texts(rewarded_df, args, dimension)
    
def parse_args():
    parser = argparse.ArgumentParser(description="Run the reward model script")
    
    # Add arguments
    parser.add_argument('--top_n', type=int, default=100, help='Number of top texts to select')
    parser.add_argument('--raw_data_dir', type=str, default='raw_data', help='Directory containing raw data')
    parser.add_argument('--reward_model_dir', type=str, default='reward_models', help='Directory containing reward models')
    parser.add_argument('--filtered_data_dir', type=str, default='filtered_data', help='Directory to store filtered data')
    parser.add_argument('--rewarded_raw_data_dir', type=str, default='rewarded_raw_data', help='Directory to store rewarded raw data')
    parser.add_argument('--torch_dtype', type=str, default='float16', help='Torch data type')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run the model on')
    parser.add_argument('--model_name', type=str, default="TheBloke/tulu-7B-fp16", help='Model name')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum length for tokenization')
    parser.add_argument('--n_evals', type=int, default=1, help='Number of evaluation runs per completion')
    parser.add_argument('--cache_dir', type=str, default='.cache', help='Directory to cache models')
    parser.add_argument('--total_number_of_texts', type=int, default=None, help='Total number of texts to process')
    parser.add_argument('--dimensions', type=str, default='P1A', help='Dimensions to process, separated by "+"')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    return parser.parse_args()
if __name__ == "__main__":
    args = parse_args()  # Parse the command-line arguments
    set_seed(args.seed)
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    if args.torch_dtype == 'float16':
        args.torch_dtype = torch.float16

    # Process dimensions (as before)
    if not args.dimensions:
        for dimension in os.listdir(args.reward_model_dir):
            print(f'Processing dimension: {dimension}')
            process_dimension(dimension, args)
    else:
        for dimension in args.dimensions.split('+'):
            print(f'Processing dimension: {dimension}')
            process_dimension(dimension, args)

        