#!/bin/bash
#SBATCH -J image_captioning_BLIP2
#SBATCH -A 
#SBATCH --gpus-per-node=8
#SBATCH --nodes=1
#SBATCH -C "thin"
#SBATCH -t 0-06:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=
#SBATCH --output 
#SBATCH --error

cd /path/to/dataset_triplets/

# Define resource variables
GPUS=8
NNODES=1
PORT=${PORT:-29503}

# Launch the distributed job
/path/to/torchrun \
    --nproc_per_node=$GPUS \
    --nnodes=$NNODES \
    --master_port=$PORT \
    image_captioning_BLIP2_distributed.py \
    --dataset /path/to/dataset_triplets/nuscenes_objects/train \
    --batch_size 16

# Merge the captions to one file
/path/to/python \
    merge_captions.py \
    --dir /path/to/dataset_triplets/nuscenes_objects/train \
    --model_id blip2