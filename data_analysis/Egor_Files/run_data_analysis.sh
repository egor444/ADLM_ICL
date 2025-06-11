#!/bin/bash
#SBATCH --job-name=data_ana
#SBATCH --output=outfiles/analysis_log.out
#SBATCH --error=outfiles/analysis_err.err
##SBATCH --partition=universe,asteroids
##SBATCH --qos=master-queuesave
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G

ml python/anaconda3
source activate adlm-icl

echo "Running analysis job with ID: $SLURM_JOB_ID"
python -u Egor_long_analysis.py 
echo "Analysis job completed with ID: $SLURM_JOB_ID"