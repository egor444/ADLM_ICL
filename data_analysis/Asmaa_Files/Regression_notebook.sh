#!/bin/bash
#SBATCH --job-name=asmaa_analysis
#SBATCH --output=outfiles/Regression_notebook_log.out
#SBATCH --error=outfiles/Regression_notebook_err.err
##SBATCH --partition=universe,asteroids
##SBATCH --qos=master-queuesave
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G

ml python/anaconda3
source activate adlm-icl

echo "Running analysis job with ID: $SLURM_JOB_ID"
python -u Regression_notebook.py
echo "Analysis job completed with ID: $SLURM_JOB_ID"
