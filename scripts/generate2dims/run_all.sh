sbatch -o slurm-%j-%x.out \
           --requeue \
           --time=infinite \
           --gres=gpu:a6000:1 \
           -c 4 \
           --mem=30G \
           --partition=sun \
           --open-mode=append \
           --wrap="bash alphas_0.1_0.1.sh"

sbatch -o slurm-%j-%x.out \
           --requeue \
           --time=infinite \
           --gres=gpu:a6000:1 \
           -c 4 \
           --mem=30G \
           --partition=sun \
           --open-mode=append \
           --wrap="bash alphas_0.5_0.5.sh"
sbatch -o slurm-%j-%x.out \
           --requeue \
           --time=infinite \
           --gres=gpu:a6000:1 \
           -c 4 \
           --mem=30G \
           --partition=sun \
           --open-mode=append \
           --wrap="bash alphas_0.05_0.05.sh"
sbatch -o slurm-%j-%x.out \
           --requeue \
           --time=infinite \
           --gres=gpu:a6000:1 \
           -c 4 \
           --mem=30G \
           --partition=sun \
           --open-mode=append \
           --wrap="bash alphas_1_1.sh"
sbatch -o slurm-%j-%x.out \
           --requeue \
           --time=infinite \
           --gres=gpu:a6000:1 \
           -c 4 \
           --mem=30G \
           --partition=sun \
           --open-mode=append \
           --wrap="bash PreferencePrompting.sh"



