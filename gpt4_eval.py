import os
import re
import json
import numpy as np
from openai import OpenAI
import torch
from glob import glob
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from utils.process import unify_pad_eos_ids, freeze_trainable_parameters
import random
import argparse
# export OPENAI_API_KEY=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
PREFERENCE_PROMPT_DICT = {
        'P1A': 'Generate a response that can be easily understood by an elementary school student.',
        'P1B': 'Generate a response that only a PhD Student in that specific field could understand.',
        'P2A': 'Generate a response that is concise and to the point without being verbose.',
        'P2B': 'Generate a response that is very informative without missing any background information.',
        'P3A': 'Generate a response that is friendly, witty, funny, and humorous, like a close friend.',
        'P3B': 'Generate a response in an unfriendly manner.'
    }
def parse_arguments():
    parser = argparse.ArgumentParser(description="Process files for preference evaluation.")
    parser.add_argument('--dataset', type=str, default='koala', help="Dataset name (e.g., 'koala').")
    parser.add_argument('--dataset_dir', type=str, default = '/share/nikola/js3673/project/personalized_alignment/Classifier-Guided/eval_generations/baselines/classifier_guided', help="Directory containing dataset files.")
    parser.add_argument('--reward_model_base_path', type=str, default = '/share/nikola/js3673/project/personalized_alignment/Classifier-Guided/data_process/reward_models', help="Base path for reward models.")
    parser.add_argument('--prompt_dir', type=str, default='/share/nikola/js3673/project/personalized_alignment/Classifier-Guided/pref_prompt', help="Directory containing prompt files.")
    parser.add_argument('--model_name', type=str, default = 'TheBloke/tulu-7B-fp16', help="Name of the base model (e.g., 'TheBloke/tulu-7B-fp16').")
    parser.add_argument('--cache_dir', type=str, default = '/share/nikola/js3673/cache', help="Directory for caching models.")
    parser.add_argument('--baseline_path', type=str, default = "/share/nikola/js3673/project/personalized_alignment/Classifier-Guided/eval_generations/baselines", help="Path to baseline files.")
    parser.add_argument('--baseline', type=str, default = 'NoPreference', help="Baseline method (e.g., 'No_preference').")
    parser.add_argument('--use_reward_model', action='store_true', help="Use reward model for evaluation (default: False).")
    parser.add_argument('--output_file', type=str, default = '/share/nikola/js3673/project/personalized_alignment/Classifier-Guided/eval_generations/output/', help="Path to store final results.")
    parser.add_argument('--gpt4_model_name', type=str, default = 'gpt-4o-2024-11-20', help="GPT-4 model name.")
    parser.add_argument('--total_dims', type=int, default = 1, help="total_nums of dims to evaluate")
    parser.add_argument('--params_file', type=str, default=None)

    return parser.parse_args()
def parse_file_info(file_path):
    # Extract dimensions from the directory name
    dims_match = re.search(r'/([^/]+)/alphas', file_path)
    if dims_match:
        dims = dims_match.group(1).split('_')
        
        
    # Extract alphas from the file name

    alphas_match = re.search(r'alphas_([\d.]+(?:_[\d.]+)*)', file_path)
    if alphas_match:
        alphas_str = alphas_match.group(1)  # Get the matched part: '0.10_0.30_0.20'
        alphas = alphas_str.split('_')

    return dims, alphas





# Function to evaluate reward using reward models
def evaluate_reward(texts, model, tokenizer, device='cuda:0'):
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, max_length=512, padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        reward = model(**inputs)[0].squeeze().tolist()
    return reward
def process_text(instruction, input_text, preference_prompt, completion):
    """
    Combine instruction, input, preference prompt, and completion into a single text.
    """
    if input_text:
        text = f"<|user|>\n{instruction} {input_text} {preference_prompt}\n<|assistant|>\n{completion}"
    else:
        text = f"<|user|>\n{instruction} {preference_prompt}\n<|assistant|>\n{completion}"
    return text


def extract_preference_from_gpt_output(output_text):
    """
    Extracts the preferred output from GPT-4's response based on the instruction format.
    Looks for the text after '### Result for example 4:'.
    """
    # Use regex to find the result line
    match = re.search(r'### Result for example 4:\s*(.*)', output_text, re.IGNORECASE)
    if match:
        result_line = match.group(1).strip().lower()
        # Normalize result
        if 'output (a)' in result_line:
            return 'Output (a)'
        elif 'output (b)' in result_line:
            return 'Output (b)'
        elif 'tie' in result_line:
            return 'TIE'
    # If the expected format is not found, return None
    return None

def evaluate_with_gpt(outputs1, outputs2, instructions, dim, args):
    with open(os.path.join(args.prompt_dir, f'{dim}.txt'), "r") as file:
        template = file.read()
    
    preferences = []
    total = len(outputs1)
    wins = 0
    losses = 0
    ties = 0



    for i, (o1, o2, ins) in enumerate(zip(outputs1, outputs2, instructions)):
        # Randomly decide the order of outputs
        outputs = [(o1, 'baseline'), (o2, 'classifier_guided')]
        random.shuffle(outputs)
        (output_a_text, model_a), (output_b_text, model_b) = outputs[0], outputs[1]
        
        # Fill in the template
        prompt = template.format(
            instruction=ins,
            output_1=output_a_text,
            output_2=output_b_text
        )
        
        # Send the prompt to GPT-4
        try:
            response = args.client.chat.completions.create(
                model= args.gpt4_model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant that evaluates text based on given instructions."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.0,
            )
            output_text = response.choices[0].message.content
            
            # Extract preference from GPT output
            preference = extract_preference_from_gpt_output(output_text)
            preferred_model = None
            # Map preference back to the original models
            if preference == 'Output (a)':
                preferred_model = model_a
            elif preference == 'Output (b)':
                preferred_model = model_b
            elif preference == 'TIE':
                preferred_model = 'TIE'
            else:
                print(f'data sample {i} without a preference')
            
            preferences.append(preferred_model)
            
            # Update win/loss/tie counts for model1
            if preferred_model == 'baseline':
                losses += 1
            elif preferred_model == 'classifier_guided':
                wins += 1
            elif preferred_model == 'TIE':
                ties += 1
            else:
                # Handle cases where preference could not be determined
                print(f"Could not determine preference for example {i+1}")
                ties += 1  # Treat as a tie or handle differently as needed
                
        except Exception as e:
            print(f"Error during GPT-4 evaluation for example {i+1}: {e}")
            preferences.append(None)
            ties += 1  # Treat errors as ties or handle differently as needed
    
    win_rate = wins / total
    lose_rate = losses / total
    tie_rate = ties / total
    
    print(f"Win rate: {win_rate*100:.2f}%")
    print(f"Lose rate: {lose_rate*100:.2f}%")
    print(f"Tie rate: {tie_rate*100:.2f}%")
    
    return preferences, win_rate, lose_rate, tie_rate




# Function to process two files
def process_files(file1, file2 ,args):
    
    dims2, alphas2 = parse_file_info(file2)
    dimensions = dims2
    # Load data from files
    with open(file1, 'r') as f:
        data1 = json.load(f)
    with open(file2, 'r') as f:
        data2 = json.load(f)
    
    # Initialize results
    results = {}
    if args.use_reward_model:
        for dim in dimensions:
            outputs1 = [process_text(entry['original_instruction'], '', PREFERENCE_PROMPT_DICT[dim], entry["output_only"]) for entry in data1]
            outputs2 = [process_text(entry['original_instruction'], '', PREFERENCE_PROMPT_DICT[dim], entry["output_only"]) for entry in data2]
            results[dim] = {}
            
            # Load reward model for the dimension
            reward_model_path = os.path.join(args.reward_model_base_path, dim, 'checkpoint-1000')
            model = AutoModelForSequenceClassification.from_pretrained(
                args.model_name, num_labels=1, torch_dtype=torch.float16, load_in_4bit=True, cache_dir=args.cache_dir
            )
            unify_pad_eos_ids(model, args.tokenizer)
            model = PeftModel.from_pretrained(model, reward_model_path)
            freeze_trainable_parameters(model)
            model.eval().to(args.device)
            # Evaluate outputs
            rewards1 = evaluate_reward(outputs1, model, args.tokenizer)
            rewards2 = evaluate_reward(outputs2, model, args.tokenizer)
            # Compute metrics
            mean_reward1 = np.mean(rewards1)
            mean_reward2 = np.mean(rewards2)
            win_rate = np.mean(np.array(rewards1) < np.array(rewards2))
            tie_rate = np.mean(np.array(rewards1) == np.array(rewards2))
            lose_rate = np.mean(np.array(rewards1) > np.array(rewards2))
            # Store results
            results[dim]['baseline_reward'] = mean_reward1
            results[dim]['mean_reward_file2'] = mean_reward2
            results[dim]['win_rate'] = win_rate
            results[dim]['tie_rate'] = tie_rate
            results[dim]['lose_rate'] = lose_rate
        return {'dims': dimensions,
                'alphas2': alphas2,
                'results': results,
                'average_win_rate': np.mean([results[dim]['win_rate'] for dim in dimensions]),
                'average_tie_rate': np.mean([results[dim]['tie_rate'] for dim in dimensions]),
                'average_lose_rate': np.mean([results[dim]['lose_rate'] for dim in dimensions]),
                }
    else:
        for dim in dimensions:
            outputs1 = [entry["output_only"] for entry in data1]
            outputs2 = [entry["output_only"] for entry in data2]
            instructions = [entry['original_instruction'] for entry in data1]
            results[dim] = {}
            # Evaluate outputs using GPT-4
            preferences, win_rate, lose_rate, tie_rate = evaluate_with_gpt(outputs1, outputs2, instructions, dim, args)
            

            # Store results
            results[dim]['win_rate'] = win_rate
            results[dim]['tie_rate'] = tie_rate
            results[dim]['lose_rate'] = lose_rate
            results[dim]['preferences'] = preferences
        return {'dims': dimensions,
                'alphas2': alphas2,
                'results': results,
                'average_win_rate': np.mean([results[dim]['win_rate'] for dim in dimensions]),
                'average_tie_rate': np.mean([results[dim]['tie_rate'] for dim in dimensions]),
                'average_lose_rate': np.mean([results[dim]['lose_rate'] for dim in dimensions]),
                }        
            
        




# Main function to process multiple files
def main():
    args = parse_arguments()

    
    args.tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    args.client = OpenAI()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'


    # Function to parse file names

    # Get list of all JSON files in the directory
    json_files = glob(os.path.join(args.dataset_dir, args.dataset, '*', 'alphas_*.json'))
    # Group files by dimensions
    file_groups = {}
    for file_path in json_files:
        dims, alphas = parse_file_info(file_path)
        dims_key = '_'.join(dims)
        if dims_key not in file_groups:
            file_groups[dims_key] = []
        file_groups[dims_key].append((alphas, file_path))

    # For each group, sort files by alphas and process
    result = []
    output_file = os.path.join(args.output_file, f'{args.dataset}/{args.baseline}_useRM_{int(args.use_reward_model)}_total_dim{args.total_dims}.json')
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            result = json.load(f)
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))   
    for dims_key, files in file_groups.items():
        # Sort files based on alphas
        files.sort(key=lambda x: x[0])
        # Process pairs of files
        for i in range(len(files)):
            if args.baseline == 'NoPreference':
                baseline_file = os.path.join(args.baseline_path, args.baseline, args.dataset,'generation.json')
            else:
                baseline_file = os.path.join(args.baseline_path, args.baseline, args.dataset, dims_key, 'generation.json')
            file2 = files[i][1]

            dims2, alphas2 = parse_file_info(file2)
            
            if len(alphas2) != args.total_dims:
                print('len alphas', len(alphas), 'total_dims',args.total_dims)

                print(f"Skipping {dims2} with {len(alphas2)} alphas")
                continue
            if any(entry["dims"] == dims2 and entry["alphas2"] == alphas2 for entry in result):
                print(f'result for dim {dims2} and alphas {alphas2} already exists, skipping....')
                continue
            print('args', args)
            print('args.params', args.params_file)
            if args.params_file:
                with open(args.params_file, 'r') as f:
                    PREFERENCE2ALPHA = json.load(f)
                if '_'.join(dims2) not in PREFERENCE2ALPHA:
                    print(f"Skipping {dims2} with {alphas2} alphas")
                    continue
                if PREFERENCE2ALPHA['_'.join(dims2)] != alphas2:
                    print(f"Skipping {PREFERENCE2ALPHA['_'.join(dims2)]} with {alphas2} alphas")
                    continue

            
            
            
            result.append(process_files(baseline_file, file2, args)) # Set to False for GPT evaluation
            # Save results to JSON file

            with open(output_file, 'w') as f:
                json.dump(result, f, indent=4)
            print(f"Results saved to {output_file}")
if __name__ == "__main__":
    main()
