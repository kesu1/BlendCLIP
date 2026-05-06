#!/bin/bash
#SBATCH -J create_nuscenes_objects
#SBATCH -A 
#SBATCH --partition=
#SBATCH --mem=512G
#SBATCH -t 1-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=
#SBATCH --output 
#SBATCH --error

cd /path/to/dataset_triplets/

module load Mambaforge
mamba activate triplets

python prepare_data_nuscenes.py \
    --dataroot /path/to/nuscenes \
    --version v1.0-trainval \
    --split val \
    --output /path/to/nuscenes_objects \
    --pts_tresh 1 \