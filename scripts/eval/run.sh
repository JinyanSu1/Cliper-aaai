sbatch -o slurm-%j-%x.out \
           --requeue \
           --time=infinite \
           --gres=gpu:a6000:1 \
           -c 4 \
           --mem=30G \
           --partition=sun \
           --open-mode=append \
           --wrap="bash eval_Nopreference1.sh"
# sbatch -o slurm-%j-%x.out \
#            --requeue \
#            --time=infinite \
#            --gres=gpu:a6000:1 \
#            -c 4 \
#            --mem=30G \
#            --partition=sun \
#            --open-mode=append \
#            --wrap="bash eval_Nopreference2.sh"
