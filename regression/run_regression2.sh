#!/bin/bash
#SBATCH --job-name=regression
#SBATCH --output=outfiles4/tregression_test_log.out
#SBATCH --error=outfiles4/tregression_test_err.err
##SBATCH --partition=universe,asteroids
##SBATCH --qos=master-queuesave
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

ml python/anaconda3



echo "Starting Regression job with SLURM_JOB_ID: $SLURM_JOB_ID"

# Set Python path to find modules
export PYTHONPATH=/vol/miltank/projects/practical_sose25/in_context_learning:$PYTHONPATH

python -u train_regression.py

echo "Regression job completed with SLURM_JOB_ID: $SLURM_JOB_ID"
