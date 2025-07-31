#!/bin/bash
#SBATCH --job-name=gpt2_test
#SBATCH --output=outputs/gpt2_imcontext_regression.out
#SBATCH --error=outputs/gpt2_imcontext_regression.err
##SBATCH --partition=universe,asteroids
##SBATCH --qos=master-queuesave
#SBATCH --time=08:20:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

ml python/anaconda3
#source activate adlm-icl
#nvidia-smi
echo "Running gpt2 regression script"
python -u /vol/miltank/projects/practical_sose25/in_context_learning/regression/models/gpt2_regressor.py