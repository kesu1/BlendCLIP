#!/bin/bash
#SBATCH -J create_truckscenes_objects
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

python prepare_data_truckscenes.py \
    --dataroot /path/to/man-truckscenes \
    --version v1.0-trainval \
    --split val \
    --output /path/to/official_eval \
    --pts_tresh 1 \