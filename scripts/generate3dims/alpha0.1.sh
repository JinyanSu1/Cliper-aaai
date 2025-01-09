#!/bin/bash

# Arrays of values to iterate over
alphas=("0.1" "0.1" "0.1")  # Add multiple alpha values here
choices=("A" "B")           # Choose A or B for each index
use_personalized_prompts=(True)
mixed_prompts=(True)
output_dir='../../eval/baselines/classifier_guided'
datasets=("alpaca_evaluation100" "alpaca_test100")   # Corrected to iterate over each dataset

# Generate all 8 combinations of A or B for (1, 2, 3)
for choice1 in "${choices[@]}"; do
    for choice2 in "${choices[@]}"; do
        for choice3 in "${choices[@]}"; do
            # Construct preference symbols for the current combination
            preference_symbols=("P1${choice1}" "P2${choice2}" "P3${choice3}")

            # Combine preference symbols for the filename
            preference_symbols_combined=$(IFS=_; echo "${preference_symbols[*]}")
            alphas_combined=$(IFS=_; echo "${alphas[*]}")

            for use_personalized_prompt in "${use_personalized_prompts[@]}"; do
                for mixed_prompt in "${mixed_prompts[@]}"; do
                    for dataset in "${datasets[@]}"; do  # Added loop for datasets
                        prompt_flag=$( [ "$use_personalized_prompt" = "True" ] && echo 1 || echo 0 )
                        mixed_prompt_flag=$( [ "$mixed_prompt" = "True" ] && echo 1 || echo 0 )

                        # Construct the Python command
                        python_cmd="python ../../generate3dims.py \
                            --cache_dir '/share/nikola/js3673/cache' \
                            --tokenizer_model 'TheBloke/tulu-7B-fp16' \
                            --classifier_tokenizer_model 'JackFram/llama-160m' \
                            --classifier_model_path '../../models/classifier/llama160m_10000' \
                            --dataset_path '../../eval/data' \
                            --dataset ${dataset} \
                            --output_dir ${output_dir} \
                            --alphas ${alphas[*]} \
                            --preference_symbols ${preference_symbols[*]} \
                            --batch_size 8 \
                            --max_new_tokens 512 \
                            --temperature 0.1 \
                            --num_return_sequences 1 \
                            --device 'cuda:0' \
                            --seed 42"

                        # Add --use_personalized_prompt if it's True
                        if [ "$use_personalized_prompt" = "True" ]; then
                            python_cmd+=" --use_personalized_prompt"
                        fi
                        # Add --mixed_preference_prompt if it's True
                        if [ "$mixed_prompt" = "True" ]; then
                            python_cmd+=" --mixed_preference_prompt"
                        fi

                        # Execute the command
                        echo "Running: $python_cmd"
                        eval $python_cmd
                    done
                done
            done
        done
    done
done
