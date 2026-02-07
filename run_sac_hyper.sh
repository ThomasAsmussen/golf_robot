#!/bin/sh
#BSUB -q hpc
#BSUB -J sac
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=1GB]"
#BSUB -W 23:59
#BSUB -o outputs/sac_%J.out
#BSUB -e outputs/sac_%J.err
#BSUB -N

echo "Running script..."

module load cuda/11.8
module load python3/3.10.13

# Activate your env
cd ~/golf_robot
source golf_venv/bin/activate

# Run the wandb sweep agent
# --count controls how many trials this job will run
wandb agent --count 5 rl_golf/golf_robot_sac_v5/qd8lijw9