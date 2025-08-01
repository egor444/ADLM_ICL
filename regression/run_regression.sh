#!/bin/bash
#SBATCH --job-name=regression
#SBATCH --output=outfile/regression.out
#SBATCH --error=outfile/regression.log
##SBATCH --partition=universe,asteroids
##SBATCH --qos=master-queuesave
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

ml python/anaconda3



echo "Starting Regression job with SLURM_JOB_ID: $SLURM_JOB_ID"

python -u regression/train_regression.py

echo "Regression job completed with SLURM_JOB_ID: $SLURM_JOB_ID"
