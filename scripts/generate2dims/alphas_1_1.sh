#!/bin/bash

# Arrays of combinations and options
combinations=("1_2" "1_3" "2_3")  # All 2-dimension combinations
choices=("A" "B")                # Choose A or B for each dimension
datasets=("alpaca_evaluation100" "alpaca_test100")  # Datasets
alphas=("1" "1")             # Scaling factors for 2 dimensions
output_dir='../../eval/baselines/classifier_guided'
use_personalized_prompts=(True)
mixed_prompts=(True)

# Iterate through all 2-dimension combinations
for combination in "${combinations[@]}"; do
    for choice1 in "${choices[@]}"; do
        for choice2 in "${choices[@]}"; do
            # Construct preference symbols for the selected dimensions
            preference_symbols=("P${combination:0:1}${choice1}" "P${combination:2:1}${choice2}")

            # Combine preference symbols for filenames/logging
            preference_symbols_combined=$(IFS=_; echo "${preference_symbols[*]}")
            alphas_combined=$(IFS=_; echo "${alphas[*]}")

            for use_personalized_prompt in "${use_personalized_prompts[@]}"; do
                for mixed_prompt in "${mixed_prompts[@]}"; do
                    for dataset in "${datasets[@]}"; do
                        prompt_flag=$( [ "$use_personalized_prompt" = "True" ] && echo 1 || echo 0 )
                        mixed_prompt_flag=$( [ "$mixed_prompt" = "True" ] && echo 1 || echo 0 )

                        # Construct the Python command for generating text
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
                            --device 'cuda' \
                            --seed 42"

                        # Add --use_personalized_prompt if True
                        if [ "$use_personalized_prompt" = "True" ]; then
                            python_cmd+=" --use_personalized_prompt"
                        fi
                        # Add --mixed_preference_prompt if True
                        if [ "$mixed_prompt" = "True" ]; then
                            python_cmd+=" --mixed_preference_prompt"
                        fi

                        # Log and execute the command
                        echo "Running: $python_cmd"
                        eval $python_cmd
                    done
                done
            done
        done
    done
done
