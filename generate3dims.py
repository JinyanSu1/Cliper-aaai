import json
import os
import argparse
from typing import Optional, List

import torch
from datasets import load_dataset
from rich.console import Console
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer, GenerationConfig,
                          set_seed)
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

from utils.custom_class import LlamaForMultiClassTokenClassification

# Initialize console for logging
console = Console(force_terminal=True)


class GuidedLogitsProcessor(LogitsProcessor):
    """
    Custom LogitsProcessor that guides the generation using a classifier model.
    """

    def __init__(
        self,
        alphas: List[float],
        model_classifier,
        preference_dims: List[int],
        use_cache: Optional[bool] = True,
    ):
        """
        Initializes the GuidedLogitsProcessor.

        Args:
            alpha (float): The scaling factor for the classifier probabilities.
            model_classifier: The classifier model to guide the generation.
            preference_dim (int): The dimension of the preference to focus on.
            use_cache (bool, optional): Whether to use cached past key values in the classifier. Defaults to True.
        """
        self.alphas = alphas
        self.model_classifier = model_classifier
        self.preference_dims = preference_dims
        self.classifier_state = {
            "input_ids": None,
            "attention_mask": None,
            "use_cache": use_cache,
            "past_key_values": None,
            "first_pass": True
        }

    def get_classifier_logits(self, input_ids):
        """
        Computes the classifier logits for the given input_ids.

        Args:
            input_ids (torch.LongTensor): The input IDs.

        Returns:
            torch.FloatTensor: The classifier logits.
        """
        if self.classifier_state["first_pass"]:
            # Initialize input_ids and attention_mask if first pass
            self.classifier_state["input_ids"] = input_ids
            self.classifier_state["attention_mask"] = torch.ones_like(input_ids, dtype=torch.long)
            input_ids = self.classifier_state["input_ids"]
            attention_mask = self.classifier_state["attention_mask"]
            self.classifier_state["first_pass"] = False
        else:
            # Update attention_mask and input_ids
            attention_mask = torch.cat(
                [
                    self.classifier_state["attention_mask"],
                    torch.ones_like(input_ids[:, -1:], dtype=torch.long),
                ],
                dim=1,
            )
            if not self.classifier_state["use_cache"]:
                input_ids = torch.cat([self.classifier_state["input_ids"], input_ids[:, -1:]], dim=1)
            else:
                input_ids = input_ids[:, -1:]
            self.classifier_state["input_ids"] = input_ids
            self.classifier_state["attention_mask"] = attention_mask

        with torch.no_grad():
            out = self.model_classifier(
                input_ids,
                attention_mask=attention_mask,
                use_cache=self.classifier_state["use_cache"],
                past_key_values=self.classifier_state["past_key_values"],
            )
        self.classifier_state["past_key_values"] = out.get("past_key_values", None)
        return out.logits

    def __call__(self, input_ids, basemodel_logits, tokenized_inputs_classifier):
        """
        Modifies the logits during generation.

        Args:
            input_ids (torch.LongTensor): The input IDs.
            basemodel_logits (torch.FloatTensor): The logits from the base model.

        Returns:
            torch.FloatTensor: The modified logits.
        """
        # Get classifier logits and convert to probabilities
        classifier_logits = self.get_classifier_logits(tokenized_inputs_classifier)
        classifier_probs = torch.softmax(classifier_logits, dim=-1)
        classifier_probs = classifier_probs[:, -1, :, :]  # Shape: [batch_size, vocab_size]

        # Get base model probabilities


        # Combine probabilities using the formula
        combined_classifier_probs = torch.ones_like(classifier_probs[:, :, 0])  # Shape: [batch_size, vocab_size]
        for alpha, preference_dim in zip(self.alphas, self.preference_dims):
            # Get classifier probabilities for the specific preference dimension
            preference_probs = classifier_probs[:, :, preference_dim]  # Shape: [batch_size, vocab_size]
            # Adjust the probabilities using alpha
            combined_classifier_probs *= preference_probs ** alpha
        basemodel_probs = torch.softmax(basemodel_logits, dim=-1)

        combined_probs = (torch.cat([combined_classifier_probs, torch.zeros_like(combined_classifier_probs[:,0].unsqueeze(1))], dim=1)) * basemodel_probs
        # Exclude specific tokens (e.g., token with index 30166)
        combined_probs[:, 30166] = 0.0  # Adjust token index as needed
        combined_probs[:, -1] = 0.0  # Exclude last token if needed

        # Normalize the probabilities
        combined_probs = combined_probs / combined_probs.sum(dim=-1, keepdim=True)

        # Convert probabilities back to logits
        # epsilon = 1e-12  # Small constant to prevent log(0)
        combined_logits = torch.log(combined_probs)
        combined_logits[:, 30166] = -float('inf')  # Ensure the excluded token has -inf logit
        combined_logits[:, -1] = -float('inf')
        return combined_logits


def guided_generation(
    input_entries,
    tokenized_inputs_classifier, 
    model_tulu,
    model_classifier,
    alphas,
    tokenizer,
    generation_config,
    preference_dims,
):
    """
    Generates text using the base model guided by the classifier model.

    Args:
        input_entries (dict): Dictionary containing 'input_ids' and 'attention_mask'.
        model_tulu: The base language model.
        model_classifier: The classifier model.
        alpha (float): Scaling factor for the classifier probabilities.
        tokenizer: The tokenizer.
        generation_config: The generation configuration.
        preference_dim (int): The preference dimension to focus on.

    Returns:
        list: Generated text outputs.
    """
    # Initialize the guided logits processor
    logits_processor_list = LogitsProcessorList([
        GuidedLogitsProcessor(alphas, model_classifier, preference_dims),
    ])

    with torch.no_grad():
        output = model_tulu.generate(
            input_ids=input_entries['input_ids'],
            attention_mask=input_entries['attention_mask'],
            logits_processor=[lambda input_ids, basemodel_logits: logits_processor_list[0](input_ids, basemodel_logits, tokenized_inputs_classifier['input_ids'])],
            generation_config=generation_config
        )

    # Decode the generated text
    text = tokenizer.batch_decode(output, skip_special_tokens=True)
    return text


def generate_prompt_tulu(personalized_instruction: str, instruction: str, input_ctxt: str = None) -> str:
    """
    Generates the prompt for the Tulu model.

    Args:
        personalized_instruction (str): The personalized instruction to include in the prompt.
        instruction (str): The main instruction.
        input_ctxt (str, optional): Additional input context. Defaults to None.

    Returns:
        str: The generated prompt.
    """
    if not input_ctxt:
        return f"<|user|>\n{instruction} {personalized_instruction}\n<|assistant|>\n"
    else:
        return f"<|user|>\n{instruction} {input_ctxt} {personalized_instruction}\n<|assistant|>\n"


def parse_args():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Guided text generation with classifier guidance.")

    # Model and tokenizer paths
    parser.add_argument('--cache_dir', type=str, default='cache', help='Cache directory for models.')
    parser.add_argument('--tokenizer_model', type=str, default='TheBloke/tulu-7B-fp16', help='Tokenizer model name or path.')
    parser.add_argument('--classifier_tokenizer_model', type=str, default='JackFram/llama-160m', help='Classifier tokenizer model name or path.')
    parser.add_argument('--classifier_model_path', type=str, required=True, help='Path to the trained classifier model.')

    # Data paths
    parser.add_argument('--dataset_path', type=str, default = 'data/processed/top1', help='Path to the input data JSON file.')
    parser.add_argument('--dataset', type=str, default = 'koala', help='name of the dataset')
    parser.add_argument('--output_dir', type=str, default='eval_generations', help='Directory to save the output JSON file.')

    # Generation parameters
    parser.add_argument('--alphas', nargs='+', type=float, required=True, help='Scaling factors for classifier probabilities.')
    parser.add_argument('--preference_symbols', nargs='+', type=str, required=True, choices=['P1A', 'P2A', 'P3A', 'P1B', 'P2B', 'P3B'], help='Preference symbols to use.')
    
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for data loading.')
    parser.add_argument('--max_new_tokens', type=int, default=512, help='Maximum number of new tokens to generate.')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature for generation.')
    parser.add_argument('--num_return_sequences', type=int, default=1, help='Number of sequences to return.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the models on.')
    parser.add_argument('--use_personalized_prompt', action='store_true', help='Whether to use personalized prompts.(For basemodel only)')
    # Miscellaneous
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--mixed_preference_prompt', action='store_true', help='Whether to use personalized prompts.(For basemodel only)')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Set random seed for reproducibility
    set_seed(args.seed)


    # Initialize console for logging
    console = Console(force_terminal=True)

    # Constants and configuration
    PREFERENCE_PROMPT_DICT = {
        'P1A': 'Generate a response that can be easily understood by an elementary school student.',
        'P1B': 'Generate a response that only a PhD Student in that specific field could understand.',
        'P2A': 'Generate a response that is concise and to the point without being verbose.',
        'P2B': 'Generate a response that is very informative without missing any background information.',
        'P3A': 'Generate a response that is friendly, witty, funny, and humorous, like a close friend.',
        'P3B': 'Generate a response in an unfriendly manner.'
    }
    SYMBOL_TO_DIM = {'P1A': 0,
                    'P1B': 1,
                    'P2A': 2,
                     'P2B': 3,
                     'P3A': 4,
                     'P3B': 5
                     }

    # Load tokenizers
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model, use_fast=False)
    tokenizer_classifier = AutoTokenizer.from_pretrained(args.classifier_tokenizer_model, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # Use left-padding for decoder-only models

    # Load models
    model_tulu = AutoModelForCausalLM.from_pretrained(
        args.tokenizer_model,
        load_in_8bit=True,
        cache_dir=args.cache_dir,
        device_map='auto',
    )
    model_classifier = LlamaForMultiClassTokenClassification.from_pretrained(
        args.classifier_model_path
    )
    model_classifier.to(args.device)
    model_classifier.eval()

    # Prepare output directory
    if not os.path.exists(os.path.dirname(args.output_dir)):
        os.makedirs(os.path.dirname(args.output_dir))


    dataset = load_dataset("json", split = 'train', data_files=os.path.join(args.dataset_path, f'{args.dataset}.json'))
        
    # dataset = load_dataset("json", split = 'train', data_files=args.data_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)

    # Get personalized prompt and preference dimension
    # Generate combined personalized prompt
    personalized_prompts = [PREFERENCE_PROMPT_DICT[symbol] for symbol in args.preference_symbols]
    combined_personalized_prompt = ' '.join(personalized_prompts)

    
    preference_dims = [SYMBOL_TO_DIM[symbol] for symbol in args.preference_symbols]
    if len(args.alphas) != len(preference_dims):
        raise ValueError("The number of alphas must match the number of preference symbols.")
    # Generation configuration
    generation_config = GenerationConfig(
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        temperature=args.temperature,
        do_sample=True,
        num_return_sequences=args.num_return_sequences,
        max_new_tokens=args.max_new_tokens,
    )
    
    results = []
    indx = 0
    args.use_personalized_prompt = bool(args.use_personalized_prompt)

    for batch in tqdm(dataloader):
        input_prompts = []
        for instruction in batch['prompt']:
            if args.use_personalized_prompt:
                prompt = generate_prompt_tulu(combined_personalized_prompt, instruction)
            else:
                prompt = generate_prompt_tulu('', instruction)
            input_prompts.append(prompt)

        
        
        tokenized_inputs = tokenizer(
            input_prompts,
            max_length=512,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        tokenized_inputs = {key: value.to(args.device) for key, value in tokenized_inputs.items()}
        if args.mixed_preference_prompt:
            tokenized_inputs_classifier = tokenizer(
                batch['prompt'],
                max_length=512,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            tokenized_inputs_classifier = {key: value.to(args.device) for key, value in tokenized_inputs_classifier.items()}
        else:
            tokenized_inputs_classifier = tokenized_inputs
            
        
        
        outputs_text = guided_generation(
            tokenized_inputs,
            tokenized_inputs_classifier, 
            model_tulu,
            model_classifier,
            args.alphas,
            tokenizer,
            generation_config,
            preference_dims
        )

        for input_text, output_text, original_instruction in zip(input_prompts, outputs_text, batch['prompt']):
            output_only = output_text[len(input_text):].strip()
            console.print(f'INPUT: {input_text}')
            console.print(f'OUTPUT: {output_text}')
            entry = {
                "id": indx,
                "prompt": input_text,
                "output": output_text,
                "original_instruction": original_instruction,
                "output_only": output_only
            }
            results.append(entry)
            indx += 1

    # Save results to JSON file
    preference_symbols_str = "_".join(args.preference_symbols)
    alphas_str = "_".join([f"{alpha:.2f}" for alpha in args.alphas])
    
    if sum(args.alphas) == 0:
        if args.use_personalized_prompt:
            print('Personalized Prompting baseline')
            output_file_name = os.path.join(args.output_dir,args.dataset, preference_symbols_str, f"generation.json")
        else:
            print('No Prompting baseline')
            output_file_name = os.path.join(args.output_dir, args.dataset, "generation.json")
        
    else:
        output_file_name = os.path.join(
        args.output_dir,
        args.dataset,
        preference_symbols_str,  # Directory based on preference symbols
        f"alphas_{alphas_str}_PP{int(args.use_personalized_prompt)}_MP{int(args.mixed_preference_prompt)}.json"
)

    if not os.path.exists(os.path.dirname(output_file_name)):
        os.makedirs(os.path.dirname(output_file_name))
    with open(output_file_name, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
