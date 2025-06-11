#!/bin/bash
#SBATCH --job-name=sm_optim_trials
#SBATCH --output=outputs/simple_model_training.out
#SBATCH --error=outputs//simple_model_training.err
##SBATCH --partition=universe,asteroids
##SBATCH --qos=master-queuesave
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

ml python/anaconda3
source activate adlm-icl

echo "Running training script"
python -u train_model.py