#!/bin/bash
#SBATCH --job-name=reg2
#SBATCH --output=outfiles2/reg2.out
#SBATCH --error=outfiles2/reg2.err
##SBATCH --partition=universe,asteroids
##SBATCH --qos=master-queuesave
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

ml python/anaconda3

echo "Starting Regression job with SLURM_JOB_ID: $SLURM_JOB_ID"

# Set Python path to find modules
export PYTHONPATH=/vol/miltank/projects/practical_sose25/in_context_learning:$PYTHONPATH

python -u train_regression_adv_selection.py

echo "Regression job completed with SLURM_JOB_ID: $SLURM_JOB_ID"