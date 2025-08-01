#!/bin/bash
#SBATCH --job-name=regression_test
#SBATCH --output=outputs/gpt2_classification.out
#SBATCH --error=outputs/gpt2_classification.err
##SBATCH --partition=universe,asteroids
##SBATCH --qos=master-queuesave
#SBATCH --time=06:20:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

ml python/anaconda3
#source activate adlm-icl
#nvidia-smi
echo "Running gpt2 classification testing script"
python -u tabpfn_test.py