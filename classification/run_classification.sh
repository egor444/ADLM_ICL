#!/bin/bash
#SBATCH --job-name=classification
#SBATCH --output=outfile/classification.out
#SBATCH --error=outfile/classification.log
##SBATCH --partition=universe,asteroids
##SBATCH --qos=master-queuesave
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

ml python/anaconda3

echo "Starting classification job with SLURM_JOB_ID: $SLURM_JOB_ID"

# Set Python path to find modules
# export PYTHONPATH=/vol/miltank/projects/practical_sose25/in_context_learning:$PYTHONPATH

# cd /vol/miltank/projects/practical_sose25/in_context_learning

# Run using the module path
python -u classification/train_classification.py
#python -u train_classification.py

echo "Classification job completed with SLURM_JOB_ID: $SLURM_JOB_ID"
