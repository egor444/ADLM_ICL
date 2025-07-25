#!/bin/bash
#SBATCH --job-name=sm_optim_trials
#SBATCH --output=outputs/simple_model_training.out
#SBATCH --error=outputs//simple_model_training.err
##SBATCH --partition=universe,asteroids
##SBATCH --qos=master-queuesave
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

ml python/anaconda3

# Set Python path to find modules
export PYTHONPATH=/vol/miltank/projects/practical_sose25/in_context_learning:$PYTHONPATH

echo "Running training script"
python -u train_model.py