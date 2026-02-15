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

python3 /zhome/41/d/156422/golf_robot/src/golf_robot/ucb_bandit_0.py > log/run_ucb$(date +"%d-%m-%y")_$(date +'%H:%M:%S').log