#!/bin/bash
#SBATCH --job-name=data_ext
#SBATCH --output=logs/xtraction.out
#SBATCH --error=logs/xtraction.err
##SBATCH --partition=universe,asteroids
##SBATCH --qos=master-queuesave
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

ml python/anaconda3
#source activate adlm-icl

echo "Running extraction job with ID: $SLURM_JOB_ID"
python -u data_extraction.py --test_data_mgr
echo "Extraction job completed with ID: $SLURM_JOB_ID"