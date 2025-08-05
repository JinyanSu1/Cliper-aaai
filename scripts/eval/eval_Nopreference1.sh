


#!/bin/bash

# Set default values
DEFAULT_DATASET_DIR="../../eval/baselines/classifier_guided"
DEFAULT_REWARD_MODEL_BASE_PATH="../../models/reward_models"
DEFAULT_PROMPT_DIR="../../prompts"
DEFAULT_MODEL_NAME="TheBloke/tulu-7B-fp16"
DEFAULT_CACHE_DIR="/share/nikola/js3673/cache"
DEFAULT_BASELINE_PATH="../../eval/baselines"
DEFAULT_OUTPUT_FILE="../../eval/results"
DEFAULT_PARAMS_FILE="../../eval/params/param_RM0.json"
# DEFAULT_GPT4_MODEL_NAME="gpt-4o-2024-11-20"
DEFAULT_GPT4_MODEL_NAME="gpt-4o-mini-2024-07-18"
export OPENAI_API_KEY=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
# Datasets to process
# DATASETS=("koala")
# BASELINES=("NoPreference")
DATASETS=('koala' 'ultrafeedback')
BASELINES=('NoPreference')

# Iterate over datasets and baselines
for dataset in "${DATASETS[@]}"; do
    for baseline in "${BASELINES[@]}"; do
     for dim in {1..3}; do
        echo "Processing dataset: $dataset with baseline: $baseline"
        
        # Run the Python script
        python3 ../../gpt4_eval.py \
            --dataset "$dataset" \
            --dataset_dir "$DEFAULT_DATASET_DIR" \
            --reward_model_base_path "$DEFAULT_REWARD_MODEL_BASE_PATH" \
            --prompt_dir "$DEFAULT_PROMPT_DIR" \
            --model_name "$DEFAULT_MODEL_NAME" \
            --cache_dir "$DEFAULT_CACHE_DIR" \
            --baseline_path "$DEFAULT_BASELINE_PATH" \
            --baseline "$baseline" \
            --output_file "$DEFAULT_OUTPUT_FILE" \
            --gpt4_model_name "$DEFAULT_GPT4_MODEL_NAME" \
            --total_dims $dim 
    done
done
done