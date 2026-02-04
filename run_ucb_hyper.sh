#!/bin/sh
#BSUB -q hpc
#BSUB -J ucb_sweep
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=1GB]"
#BSUB -W 23:59
#BSUB -o outputs/ucb_sweep_%J.out
#BSUB -e outputs/ucb_sweep_%J.err

module load cuda/11.8
module load python3/3.10.13

# Activate your env
cd ~/golf_robot
source golf_venv/bin/activate

export WANDB_CONSOLE=off
export WANDB_DISABLE_CODE=true
export WANDB_START_METHOD=thread

# Run the wandb sweep agent
# --count controls how many trials this job will run
wandb agent --count 2 rl_golf/golf_robot-src_golf_robot/iig3hetd