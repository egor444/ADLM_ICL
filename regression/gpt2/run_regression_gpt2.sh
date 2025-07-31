#!/bin/bash
#SBATCH --job-name=reg2
#SBATCH --output=outfiles6/reg2.out
#SBATCH --error=outfiles6/reg2.err
##SBATCH --partition=universe,asteroids
##SBATCH --qos=master-queuesave
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

ml python/anaconda3

echo "Starting Regression job with SLURM_JOB_ID: $SLURM_JOB_ID"

# Set Python path to find modules
export PYTHONPATH=/vol/miltank/projects/practical_sose25/in_context_learning:$PYTHONPATH

python -u train_gpt2_regression.py

echo "Regression job completed with SLURM_JOB_ID: $SLURM_JOB_ID"