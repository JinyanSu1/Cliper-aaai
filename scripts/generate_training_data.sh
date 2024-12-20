
## Generate the data with tulu model
dimensions=('P1A' 'P1B' 'P2A' 'P2B' 'P3A' 'P3B')
for dimension in "${dimensions[@]}"; do
    sbatch -o slurm-%j-%x.out \
           --requeue \
           --time=infinite \
           --gres=gpu:a6000:1 \
           -c 4 \
           --mem=50G \
           --partition=sun \
           --open-mode=append \
           --wrap="python -u tulu_generate.py --dim $dimension"
done

## Get the reward from the reward model
dimensions=('P2A' 'P3A' 'P3B')

for dimension in "${dimensions[@]}"; do
    sbatch -o slurm-%j-%x.out \
           --requeue \
           --time=infinite \
           --gres=gpu:a6000:1 \
           -c 4 \
           --mem=30G \
           --partition=sun \
           --open-mode=append \
           --wrap="python -u compute_reward_for_training_data.py --dimensions $dimension"
done


## Take the top 1 according to the reward model, and store the data in "processed_data_top1" directory

## Get the reward matrix for further analysis
#!/bin/bash
dimensions=('P1A' 'P1B' 'P2A' 'P2B' 'P3A' 'P3B')

for dimension in "${dimensions[@]}"; do
    sbatch -o slurm-%j-%x.out \
           --requeue \
           --time=infinite \
           --gres=gpu:a6000:1 \
           -c 4 \
           --mem=30G \
           --partition=sun \
           --open-mode=append \
           --wrap="python -u compute_reward_matrix.py --eval_dim $dimension"
done