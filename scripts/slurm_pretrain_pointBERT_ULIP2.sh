#!/bin/bash
#SBATCH -J pretrain_PointBERT_ULIP2_reproduction
#SBATCH -A
#SBATCH --gpus-per-node=8
#SBATCH --nodes=1
#SBATCH -C "thin"
#SBATCH -t 1-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=
#SBATCH --output
#SBATCH --error

cd /path/to/repo/

module load buildenv-gcccuda/12.1.1-gcc12.3.0
module load Ninja

# Define resource variables
GPUS=8
NNODES=1
PORT=${PORT:-29503}

# Launch the distributed job
/path/to/torchrun \
    --nproc_per_node=$GPUS \
    --nnodes=$NNODES \
    --master_port=$PORT \
    main.py \
        --pretrain_dataset_name objaverse \
        --validate_dataset_name objaverse_lvis_colored \
        --validate_dataset_prompt modelnet40_64 \
        --validate_dataset_name nuscenes_objects_official \
        --validate_dataset_prompt outdoors_1 \
        --model ULIP2_PointBERT \
        --batch-size 64 \
        --lr 1e-3 \
        --lr-block 1e-3 \
        --linear-projection \
        --wd 0.1 \
        --warmup-epochs 1 \
        --npoints 8192 \
        --output-dir ./outputs/pointBERT_ULIP2_reproduction \
        --epochs 250 \
        --eval-freq 1 \
        --workers 20 \
        --wandb