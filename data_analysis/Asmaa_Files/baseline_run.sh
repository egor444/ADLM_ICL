#!/bin/bash
#SBATCH --job-name=ml_experiment
#SBATCH --output=logs/run_%j.out
#SBATCH --error=logs/run_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

module load python/3.8
module load cuda/11.2
source ~/venvs/mlenv/bin/activate

mkdir -p logs outputs/results/regression outputs/results/classification

echo "Starting regression task..."
python train_regression.py --apply_pca --user_id $USER

echo "Starting classification task..."
python train_classification.py --apply_pca --user_id $USER
