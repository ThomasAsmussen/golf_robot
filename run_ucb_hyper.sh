#!/bin/sh
#BSUB -q hpc
#BSUB -J ucb_sweep
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2GB]"
#BSUB -W 23:59
#BSUB -o outputs/ucb_sweep_%J.out
#BSUB -e outputs/ucb_sweep_%J.err
#BSUB -N

module load cuda/11.8
module load python3/3.10.13

# Activate your env
cd ~/golf_robot
source golf_venv/bin/activate

# Run the wandb sweep agent
# --count controls how many trials this job will run
wandb agent --count 5 rl_golf/golf_robot_ucb_v_5_final_tuning/4g7l7jry