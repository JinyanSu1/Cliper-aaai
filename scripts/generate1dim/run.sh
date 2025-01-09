sbatch -o slurm-%j-%x.out \
           --requeue \
           --time=infinite \
           --gres=gpu:a6000:1 \
           -c 4 \
           --mem=30G \
           --partition=sun \
           --open-mode=append \
           --wrap="bash alpha0.1.sh"

sbatch -o slurm-%j-%x.out \
           --requeue \
           --time=infinite \
           --gres=gpu:a6000:1 \
           -c 4 \
           --mem=30G \
           --partition=sun \
           --open-mode=append \
           --wrap="bash alpha0.3.sh"
sbatch -o slurm-%j-%x.out \
           --requeue \
           --time=infinite \
           --gres=gpu:a6000:1 \
           -c 4 \
           --mem=30G \
           --partition=sun \
           --open-mode=append \
           --wrap="bash alpha0.5.sh"
sbatch -o slurm-%j-%x.out \
           --requeue \
           --time=infinite \
           --gres=gpu:a6000:1 \
           -c 4 \
           --mem=30G \
           --partition=sun \
           --open-mode=append \
           --wrap="bash alpha0.05.sh"

sbatch -o slurm-%j-%x.out \
           --requeue \
           --time=infinite \
           --gres=gpu:a6000:1 \
           -c 4 \
           --mem=30G \
           --partition=sun \
           --open-mode=append \
           --wrap="bash alpha0.8.sh"
sbatch -o slurm-%j-%x.out \
           --requeue \
           --time=infinite \
           --gres=gpu:a6000:1 \
           -c 4 \
           --mem=30G \
           --partition=sun \
           --open-mode=append \
           --wrap="bash alpha0.sh"
sbatch -o slurm-%j-%x.out \
           --requeue \
           --time=infinite \
           --gres=gpu:a6000:1 \
           -c 4 \
           --mem=30G \
           --partition=sun \
           --open-mode=append \
           --wrap="bash alpha1.sh"
sbatch -o slurm-%j-%x.out \
           --requeue \
           --time=infinite \
           --gres=gpu:a6000:1 \
           -c 4 \
           --mem=30G \
           --partition=sun \
           --open-mode=append \
           --wrap="bash NoPreference.sh"