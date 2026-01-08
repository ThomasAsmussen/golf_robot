#!/bin/sh
#BSUB -q hpc
#BSUB -J sac
### number of core
#BSUB -n 1
#BSUB -R "span[hosts=1]"
### specify the memory needed
#BSUB -R "rusage[mem=1GB]"
### Number of hours needed
#BSUB -W 23:59
### added outputs and errors to files
#BSUB -o outputs/Output_%J.out
#BSUB -e outputs/Error_%J.err

echo "Running script..."

#module load cuda/11.8
module load python3/3.10.13

# Activate your env
cd ~/golf_robot
source golf_venv/bin/activate

# Run the wandb sweep agent
# --count controls how many trials this job will run
wandb agent --count 1 rl_golf/golf_robot-src_golf_robot/vvmr4b2g