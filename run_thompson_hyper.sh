#!/bin/bash
#BSUB -q hpc          # or a CPU queue if you don't use GPU
#BSUB -J thompson_sweep
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=1GB]"
#BSUB -W 23:59            # walltime (hh:mm)
#BSUB -o outputs/thompson_sweep_%J.out
#BSUB -e outputs/thompson_sweep_%J.err
#BSUB -N
# Load modules

module load cuda/11.8
module load python3/3.10.13

# Activate your env
cd ~/golf_robot
source golf_venv/bin/activate

# Run the wandb sweep agent
# --count controls how many trials this job will run
wandb agent --count 5 rl_golf/golf_robot_thompson_v1/oopa1vpb