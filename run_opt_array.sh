#!/bin/sh
#BSUB -q hpc
#BSUB -J start_end_opt[1-78]
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=500MB]"
#BSUB -W 23:59
#BSUB -o outputs/Output_%J_%I.out
#BSUB -e outputs/Error_%J_%I.err

echo "Running start/end offset optimization array job..."
echo "JobID: $LSB_JOBID  Index: $LSB_JOBINDEX"

module load python3/3.10.13

cd ~/golf_robot
source golf_venv/bin/activate

# LSF array index is 1..N; convert to 0..N-1 for --shard-idx
SHARD_IDX=$((LSB_JOBINDEX - 1))
NUM_SHARDS=78
        
# Run one shard of the grid
python src/golf_robot/planning/optimize_start_end_offset.py \
  --speed-min 1.50 --speed-max 2.0 --speed-n 6 \
  --angle-min -6.0 --angle-max 6.0 --angle-n 13 \
  --shard-idx ${SHARD_IDX} --num-shards ${NUM_SHARDS} \
  --out-dir src/golf_robot/planning/log \
  --tag speed_angle_grid
