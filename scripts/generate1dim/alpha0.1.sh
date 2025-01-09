# Arrays of values to iterate over
alphas=("0.1")  # Alpha scaling factor
choices=("A" "B")  # A or B for each dimension
use_personalized_prompts=(True)
mixed_prompts=(True)
output_dir='../../eval/baselines/classifier_guided'
datasets=("alpaca_evaluation100" "alpaca_test100")  # Datasets to evaluate on
dimensions=("1" "2" "3")
# Iterate through dimensions and choices
for dimension in "${dimensions[@]}"; do
    for choice in "${choices[@]}"; do
        # Construct the preference symbol for the current dimension and choice
        preference_symbol="P${dimension}${choice}"

        for use_personalized_prompt in "${use_personalized_prompts[@]}"; do
            for mixed_prompt in "${mixed_prompts[@]}"; do
                for dataset in "${datasets[@]}"; do
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
                        --alpha '${alphas[0]}' \
                        --preference_symbol '${preference_symbol}' \
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