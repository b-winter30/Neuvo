#!/bin/bash --login
#
#SBATCH --job-name=neuro_object # Job name
#SBATCH --account=scw1978
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=256000MB
#SBATCH --time=3-00:00
#SBATCH --output=neuro_object.out.%J
#SBATCH --error=neuro_object.err.%J
#SBATCH --mail-type=ALL

module load anaconda
source activate
conda activate tf
conda install -c anaconda cudnn
python neuro.py
