# Toward Scalable LLM Personalization: Classifier-Guided Inference without Additional Fine-Tuning

## Train the Guidance Model (Classifier)

### Prepare Data
We use the Alpaca dataset (located at `./data/Train_classifier/raw/alpaca_gpt4_10k.json`) as our base dataset. To construct the training set for the classifier, follow these steps:

#### Step 1: Generate Outputs for Each Preference Dimension
Use the script below to generate outputs for each preference dimension:

```bash
#!/bin/bash

dimensions=('P1A' 'P1B' 'P2A' 'P2B' 'P3A' 'P3B')

# Loop through each dimension and run the Python script
for dimension in "${dimensions[@]}"; do
    python -u tulu_generate.py --dim "$dimension"
done
```

The generated data will be stored in the directory: `./data/Train_classifier/processed/raw_generation`

#### Step 2: Calculate Rewards for These Generations
Run the following script to calculate rewards for the generated data:

```bash
# Array of dimensions
dimensions=('P1A' 'P1B' 'P2A' 'P2B' 'P3A' 'P3B')

# Loop through each dimension and run the Python script
for dimension in "${dimensions[@]}"; do
    python -u compute_reward_for_training_data.py --dimensions "$dimension"
done
```

#### Step 3: Select the Top-1 Output Based on Reward
Use the following Python code to process the data and keep only the highest-reward output for each entry:

<details>
<summary>Python Script to Select Top-1 Output</summary>

```python
import os
import json

# Define the input and output directories
input_dir = '/share/nikola/js3673/project/personalized_alignment/Classifier-Guided/data/processed_data'
output_dir = '/share/nikola/js3673/project/personalized_alignment/Classifier-Guided/data/processed_data_processed'

# List of folders to process
folders = ['P1A', 'P2A', 'P3A', 'P1B', 'P2B', 'P3B']

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

for folder in folders:
    input_folder_path = os.path.join(input_dir, folder)
    output_folder_path = os.path.join(output_dir, folder)

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder_path, exist_ok=True)

    # List all JSON files in the input folder
    files = [f for f in os.listdir(input_folder_path) if f.endswith('.json')]

    for file_name in files:
        input_file_path = os.path.join(input_folder_path, file_name)
        output_file_path = os.path.join(output_folder_path, file_name)

        # Read the JSON data
        with open(input_file_path, 'r') as f:
            data_list = json.load(f)

        # Ensure the data is a list
        if isinstance(data_list, list):
            processed_data_list = []

            for data in data_list:
                # Ensure 'rewards' and 'outputs' are present and have the same length
                if 'rewards' in data and 'outputs' in data and len(data['rewards']) == len(data['outputs']):
                    # Find the index of the maximum reward
                    max_reward_index = data['rewards'].index(max(data['rewards']))

                    # Keep only the output and reward with the highest reward
                    data['outputs'] = [data['outputs'][max_reward_index]]
                    data['rewards'] = [data['rewards'][max_reward_index]]

                    # Append the processed data to the list
                    processed_data_list.append(data)
                else:
                    print(f"Skipping an entry in {input_file_path}: 'rewards' and 'outputs' are mismatched or missing.")
                    continue  # Skip entries that don't meet criteria

            # Save the processed data list to the output file
            with open(output_file_path, 'w') as f:
                json.dump(processed_data_list, f, indent=4)

            print(f"Processed file: {output_file_path}")

        else:
            print(f"Skipping file {input_file_path}: Expected a list of data entries.")
```

</details>

The final dataset for training the classifier is stored in: `./data/Train_classifier/processed/top1`

### Scripts and Reproducibility

#### Generate Training Data
We provide a bash script for Steps 1 and 2 in `./scripts/generate_training_data.sh`.

#### Reproduce the Reward Matrix
To reproduce the reward matrix, run:

```bash
dimensions=('P1A' 'P1B' 'P2A' 'P2B' 'P3A' 'P3B')

# Loop through each dimension and run the Python script
for dimension in "${dimensions[@]}"; do
    python -u compute_reward_matrix.py --eval_dim "$dimension"
done
```

The resulting matrix is stored in: `./data/Train_classifier/correlation_matrix`

---

## Train the Classifier

Run the following command to train the classifier:

```bash
bash scripts/train_classifier.sh
```

---

## Generate Classifier-Guided Text for a Single Dimension

Change to the appropriate directory and run:

```bash
cd scripts/generate1dim/
bash run.sh
```

After running the script, you will get both the preference prompting and no-preference baseline outputs.

#### Replicate No-Preference Baseline for Consistency

<details>
<summary>Linux Script to Replicate No-Preference Baseline</summary>

```bash
#!/bin/bash

# List of keys (directories to create)
keys=("P1A" "P1B" "P2A" "P2B" "P3A" "P3B")

# Source file
source_file="P1A/generation.json"

# Base directory
base_dir="./eval/baselines/classifier_guided/alpaca_evaluation100"
source_file="$base_dir/$cource_file"
# Loop through each key
for key in "${keys[@]}"; do
  # Create the directory if it doesn't exist
  target_dir="$base_dir/$key"
  mkdir -p "$target_dir"

  # Copy the file into the target directory
  cp "$source_file" "$target_dir/"
done

echo "Replication completed!"
```

</details>

---

## Baseline Evaluation

To get the preference prompt baseline and no-preference prompting baseline directly from our scripts, ensure to rename the files using the following command:

```bash
find ./eval_generations/baselines/PreferencePrompting/alpaca_evaluation -type f -name "*.json" -execdir mv '{}' generation.json \;
```

---


