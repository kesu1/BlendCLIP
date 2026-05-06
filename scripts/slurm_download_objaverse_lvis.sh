#!/bin/bash
#SBATCH -J download_objaverse_lvis
#SBATCH -A
#SBATCH --partition=
#SBATCH -t 0-12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=
#SBATCH --output
#SBATCH --error

cd /path/to/data_folder

module load Mambaforge
mamba activate gdownload

gsutil -m cp -n -r "gs://sfr-ulip-code-release-research/ULIP-2/objaverse-lvis" .