#!/bin/bash
#SBATCH --job-name=data_ext
#SBATCH --output=job_out/xtraction.out
#SBATCH --error=job_out/xtraction.err
##SBATCH --partition=universe,asteroids
##SBATCH --qos=master-queuesave
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

ml python/anaconda3
#source activate adlm-icl

echo "Running extraction job with ID: $SLURM_JOB_ID"
python -u extract_age_data.py --pca_reg --pca_class --combine_rad_emb_healthy_and_cancer --create_class
echo "Extraction job completed with ID: $SLURM_JOB_ID"