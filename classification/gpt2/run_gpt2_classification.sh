#!/bin/bash
#SBATCH --job-name=gpt2_classification
#SBATCH --output=outfiles3/gpt2_classification.out
#SBATCH --error=outfiles3/gpt2_classification.err
##SBATCH --partition=universe,asteroids
##SBATCH --qos=master-queuesave
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

ml python/anaconda3

echo "Starting Classification job with SLURM_JOB_ID: $SLURM_JOB_ID"

# Set Python path to find modules
export PYTHONPATH=/vol/miltank/projects/practical_sose25/in_context_learning:$PYTHONPATH

python -u train_gpt2_classification.py

echo "Classification job completed with SLURM_JOB_ID: $SLURM_JOB_ID"