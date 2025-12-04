#!/bin/bash
#BSUB -q gpuv100          # or a CPU queue if you don't use GPU
#BSUB -J ddpg_sweep
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 04:00            # walltime (hh:mm)
#BSUB -o logs/ddpg_sweep_%J.out
#BSUB -e logs/ddpg_sweep_%J.err

# Load modules
module load python3/3.10.13
# or whatever you used when you set up the env

# Activate your env
source ~/golf_env/bin/activate
# or: conda activate golf_env

# Go to your repo
cd ~/golf_robot

# Run the wandb sweep agent
# --count controls how many trials this job will run
wandb agent --count 10 rl_golf/golf_robot-src_golf_robot_reinforcement_learning/ontklmvk