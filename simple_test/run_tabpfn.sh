#!/bin/bash
#SBATCH --job-name=tabpfn_test
#SBATCH --output=outputs/tabpfn_test.out
#SBATCH --error=outputs/tabpfn_test.err
##SBATCH --partition=universe,asteroids
##SBATCH --qos=master-queuesave
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

ml python/anaconda3
#source activate adlm-icl
#nvidia-smi
echo "Running tabpfn testing script"
python -u tabpfn_test.py