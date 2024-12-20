import torch
from transformers import LlamaTokenizer
from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import argparse
import numpy as np
import os
import json
from utils.custom_class import LlamaForMultiClassTokenClassification
import wandb

def get_model_parameters_and_grad_status(current_model):
    """
    Get the parameters and grad status of a model.

    Args:
        current_model (`torch.nn.Module`):
            The model to get the parameters and grad status from.
    """
    res_d = {}
    for name, param in current_model.named_parameters():
        res_d[name] = (param.shape, param.requires_grad)
    return res_d

def parse_args():
    parser = argparse.ArgumentParser(description="Llama Multi-Class Token Classification")
    parser.add_argument('--model_name', type=str, default="JackFram/llama-160m", help="Pretrained model name")
    parser.add_argument('--cache_dir', type=str, default="/share/nikola/js3673/cache",
                        help="Cache directory for pretrained models")
    parser.add_argument('--num_classes', type=int, default=6, help="Number of classes for classification")
    parser.add_argument('--data_dir', type=str, default='/share/nikola/js3673/project/personalized_alignment/Classifier-Guided/data/processed_data_top1',
                        help="Directory containing JSON data files")
    parser.add_argument('--output_dir', type=str, default="./results_llama",
                        help="Output directory for training results")
    parser.add_argument('--log_dir', type=str, default='./logs', help="Directory for logging")
    parser.add_argument('--train_batch_size', type=int, default=8, help="Batch size for training")
    parser.add_argument('--eval_batch_size', type=int, default=8, help="Batch size for evaluation")
    parser.add_argument('--learning_rate', type=float, default=2e-5, help="Learning rate")
    parser.add_argument('--num_epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument('--logging_filename', type=str, default='logging_file.txt', help="logging file name")
    parser.add_argument('--eval_steps', type=int, default=50, help="Evaluate every n steps")
    parser.add_argument('--logging_steps', type=int, default=100, help="Log every n steps")
    parser.add_argument('--save_steps', type=int, default=100, help="Save model every n steps")
    parser.add_argument('--report_to', type=str, default='wandb', help="Report results to Weights & Biases")
    parser.add_argument('--wandb_run_name', type=str, default='temp', help="Weights & Biases run name")
    parser.add_argument('--finalmodel_dir', type=str, default='./final_model',
                        help="Directory for saving the final model")
    parser.add_argument('--result_dir', type=str, default='./Classifier_result', help="result_dir")
    parser.add_argument('--use_preference_prompt', type=str, default='False', help="Use preference prompt")
    return parser.parse_args()

def map_labels(data):
    # Mapping preference prompts to integer labels
    preference_prompts = [
        'Generate a response that can be easily understood by an elementary school student.',
        'Generate a response that only a PhD Student in that specific field could understand.',
        'Generate a response that is concise and to the point without being verbose.',
        'Generate a response that is very informative without missing any background information.',
        'Generate a response that is friendly, witty, funny, and humorous, like a close friend.',
        'Generate a response in an unfriendly manner.'
    ]
    label_to_int = {pref_prompt: idx for idx, pref_prompt in enumerate(preference_prompts)}
    for item in data:
        if 'preference_prompt' in item and item['preference_prompt'] in label_to_int:
            item['label'] = label_to_int[item['preference_prompt']]
        else:
            print(f"Unknown preference prompt: {item.get('preference_prompt', 'None')}")
            item['label'] = -1  # Assign -1 for unknown labels
    # Filter out entries with unknown labels
    data = [item for item in data if item['label'] != -1]
    return data

def tokenize_data(data, tokenizer, use_preference_prompt=True):
    tokenized_data = []
    a = 1
    for sample in data:
        label = sample['label']
        instruction = sample['instruction']
        input_text = sample.get('input', '')
        if use_preference_prompt:
            personalized_prompt = sample['preference_prompt']
            if input_text:
                prompt = f"{instruction}\n{input_text}\n{personalized_prompt}"
            else:
                prompt = f"{instruction}\n{personalized_prompt}"
        else:
            if input_text:
                prompt = f"{instruction}\n{input_text}"
            else:
                prompt = instruction

        response = sample['outputs'][0]  # Since we have only one output now

        text = f"<|user|>\n{prompt}\n<|assistant|>\n{response}"
        if a == 1:
            print(text)
            a = a + 1
            print('use_preference_prompt', use_preference_prompt)

        # Tokenize the concatenated text
        encoding = tokenizer(
            text,
            return_tensors='pt',
            max_length=512,  # Adjust max_length as needed
            truncation=True,
            padding='max_length'
        )

        # Prepare labels for language modeling (predict the next token in the sequence)
        input_ids = encoding.input_ids.squeeze(0).tolist()  # Convert to list
        attention_mask = encoding.attention_mask.squeeze(0).tolist()  # Convert to list
        labels = encoding.input_ids.clone().squeeze(0).tolist()  # Copy input_ids for labels

        prompt_len = len(tokenizer.encode(prompt))
        tokenized_data.append({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,  # Language model labels (shifted in forward pass)
            'class_labels': label,  # Token-level class label
            'prompt_len': prompt_len
        })
    return tokenized_data

# Define a function to compute metrics during evaluation
def compute_metrics(eval_pred):
    return {
        'average_acc': eval_pred.predictions[0].mean(),
        'loss_last_token': eval_pred.predictions[2].mean(),
    }

def main():
    args = parse_args()
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    # Logging setup
    logging.basicConfig(filename=os.path.join(args.log_dir, args.logging_filename),
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    run = wandb.init(
        project="multi-obj",         # Replace with your project name
        entity="coactivev2",           # Replace with your WandB organization or username
        name=f"{args.wandb_run_name}"                 # Provide a descriptive name for the run
    )

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LlamaForMultiClassTokenClassification.from_pretrained(args.model_name, num_labels=args.num_classes)
    model_weight_d = get_model_parameters_and_grad_status(model)
    # for k, v in model_weight_d.items():
    #     print(k, v[0], v[1])
    # print('total number of blocks:', len(model_weight_d))
    model.to(device)

    # Initialize the tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # Use left-padding for decoder-only models

    data = []
    # Loop through all files in the directory
    for root, dirs, files in os.walk(args.data_dir):
        for filename in files:
            if filename.endswith(".json"):  # Only process .json files
                file_path = os.path.join(root, filename)

                # Open and read the JSON file
                with open(file_path, 'r') as json_file:
                    data.extend(json.load(json_file))

    # Map labels based on preference prompts
    data = map_labels(data)

    # Tokenize the dataset
    if args.use_preference_prompt == 'False':
        args.use_preference_prompt = False
    elif args.use_preference_prompt == 'True':
        args.use_preference_prompt = True
    else:
        raise ValueError('Invalid value for use_preference_prompt. Must be True or False.')

    tokenized_data = tokenize_data(data, tokenizer, args.use_preference_prompt)

    # Assuming tokenized_data is your tokenized dataset
    data_df = pd.DataFrame(tokenized_data)
    data_df = data_df

    # Step 1: Split the dataset using train_test_split (80% train, 20% eval)
    train_data, eval_data = train_test_split(data_df, test_size=0.1, random_state=42)
    # test_data, eval_data = train_test_split(test_data, test_size=0.5, random_state=42)
    # Step 2: Convert to Hugging Face Datasets
    train_dataset = Dataset.from_pandas(train_data)
    eval_dataset = Dataset.from_pandas(eval_data)
    # test_dataset = Dataset.from_pandas(test_data)
    train_num_batches = len(train_dataset) // args.train_batch_size
    eval_num_batches = len(eval_dataset) // args.eval_batch_size

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    # custom training loop
    global_step = 0
    best_eval_loss= float('inf')
    result = []
    for epoch in range(args.num_epochs):
        model.train()
        
        for i in range(train_num_batches):
            global_step += 1
            batch_data = train_dataset[i * args.train_batch_size: (i + 1) * args.train_batch_size]
            input_ids = torch.Tensor(batch_data['input_ids']).long().to(device)
            attention_mask = torch.Tensor(batch_data['attention_mask']).long().to(device)
            labels = torch.Tensor(batch_data['labels']).long().to(device)
            class_labels = torch.Tensor(batch_data['class_labels']).long().to(device)
            prompt_len = torch.Tensor(batch_data['prompt_len']).long().to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                            class_labels=class_labels, prompt_len=prompt_len, return_dict=True)
            loss = outputs.loss
            average_acc = outputs.average_acc
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(
                f'Epoch: {epoch}, Step: {i}, Training Loss: {loss.item()}, Training Accuracy: {average_acc.item()}')
            run.log({'Training Loss': loss.item(), 'Average Training Accuracy': average_acc.item()}, step=global_step)
            if global_step % args.eval_steps == 0:
                model.eval()
                eval_losses = []
                eval_accs = []
                per_class_acc_eval = {cls: 0 for cls in range(args.num_classes)}
                per_class_total_eval = {cls: 0 for cls in range(args.num_classes)}
                with torch.no_grad():
                    for j in range(eval_num_batches):
                        batch_data = eval_dataset[j * args.eval_batch_size: (j + 1) * args.eval_batch_size]
                        input_ids = torch.Tensor(batch_data['input_ids']).long().to(device)
                        attention_mask = torch.Tensor(batch_data['attention_mask']).long().to(device)
                        labels = torch.Tensor(batch_data['labels']).long().to(device)
                        class_labels = torch.Tensor(batch_data['class_labels']).long().to(device)
                        prompt_len = torch.Tensor(batch_data['prompt_len']).long().to(device)
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                            class_labels=class_labels, prompt_len=prompt_len, return_dict=True)
                        loss = outputs.loss
                        average_acc = outputs.average_acc
                        eval_losses.append(loss.item())
                        eval_accs.append(average_acc.item())
                        logits = outputs.logits
                        shift_logits = logits[:, :-1, :, :].contiguous()
                        shift_labels = labels[:, 1:].contiguous()
                        shift_attention_mask = attention_mask[:, 1:].contiguous()
                        text_len = torch.sum(shift_attention_mask, dim=1)
                        text_start = shift_logits.size(1) - text_len + prompt_len - 1  # Adjusted indices
                        batch_size = input_ids.size(0)
                        for idx in range(batch_size):
                            start_idx = text_start[idx].item()
                            end_idx = shift_logits.size(1)
                            token_logits = shift_logits[idx, start_idx:end_idx, :, :]  # Shape: [seq_len, vocab_size, num_classes]
                            token_labels = shift_labels[idx, start_idx:end_idx]  # Shape: [seq_len]

                            extracted_logits = token_logits[torch.arange(token_logits.size(0)), token_labels]  # Shape: [seq_len, num_classes]

                            predicted_classes = extracted_logits.argmax(dim=-1)  # Shape: [seq_len]
                            true_class_labels = class_labels[idx].repeat(predicted_classes.size(0))  # Shape: [seq_len]

                            correct_predictions = (predicted_classes == true_class_labels).float().mean()


                            # Update per-class counts
                            cls = class_labels[idx].item()
                            per_class_acc_eval[cls] += correct_predictions.item()
                            per_class_total_eval[cls] += 1
                    per_class_accuracy_eval = {
                        cls: per_class_acc_eval[cls] / per_class_total_eval[cls] if per_class_total_eval[cls] > 0 else 0
                        for cls in range(args.num_classes)
                    }
                
                    
                
                print(
                    f'Epoch: {epoch}, Step: {i}, Evaluation Loss: {np.mean(eval_losses)}, Evaluation Accuracy: {np.mean(eval_accs)}')
                run.log({'Evaluation Loss': np.mean(eval_losses), 'Average Evaluation Accuracy': np.mean(eval_accs)},
                        step=global_step)
                for cls in per_class_accuracy_eval:
                    run.log({f"Evaluation Accuracy Class {cls}": per_class_accuracy_eval[cls]}, step=global_step)
                result.append({'golbal_step': global_step, 
                               'eval_loss': np.mean(eval_losses), 
                               'eval_acc': np.mean(eval_accs),
                                 'per_class_accuracy_eval': [acc for acc in per_class_accuracy_eval.values()],
                               })
                if not os.path.exists(args.result_dir):
                    os.makedirs(args.result_dir)
                with open(os.path.join(args.result_dir, 'result.json'), 'w') as f:
                    json.dump(result, f)
                # After calculating evaluation metrics
                eval_loss = np.mean(eval_losses)  # Calculate mean evaluation loss for this step
                if global_step % args.save_steps == 0:
                    if not os.path.exists(args.output_dir):
                            os.makedirs(args.output_dir)
                    model.save_pretrained(os.path.join(args.output_dir, f"model_{global_step}"))
                    tokenizer.save_pretrained(os.path.join(args.output_dir, f"model_{global_step}"))
                # Check if this is the best model so far
                if eval_loss < best_eval_loss:  # Or `np.mean(eval_accs) > best_eval_accuracy` for accuracy
                    best_eval_loss = eval_loss
                    best_save_path = args.finalmodel_dir

                    # Save the best model
                    if not os.path.exists(best_save_path):
                        os.makedirs(best_save_path)
                    model.save_pretrained(best_save_path)
                    tokenizer.save_pretrained(best_save_path)

                    print(f"New best model saved at {best_save_path} with evaluation loss {best_eval_loss}")

                model.train()

if __name__ == "__main__":
    main()
