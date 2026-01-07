#!/bin/sh
#BSUB -q hpc
#BSUB -J sac
### number of core
#BSUB -n 1
#BSUB -R "span[hosts=1]"
### specify that all cores should be on the same host
#BSUB -gpu "num=1:mode=exclusive_process"
### specify the memory needed
#BSUB -R "rusage[mem=1GB]"
### Number of hours needed
#BSUB -W 23:59
### added outputs and errors to files
#BSUB -o outputs/Output_%J.out
#BSUB -e outputs/Error_%J.err

echo "Running script..."

module load cuda/11.8
module load python3/3.10.13
source golf_venv/bin/activate
python3 /zhome/41/d/156422/golf_robot/src/golf_robot/reinforcement_learning/ddpg.py > log/run_ddpg$(date +"%d-%m-%y")_$(date +'%H:%M:%S').log