#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuA100 
#SBATCH --time=1:30:00
#SBATCH --job-name=factcheck_train
#SBATCH --output=factcheck_train.out
 
# Activate environment
uenv verbose cuda-12.2.0 cudnn-12.x-8.8.0
uenv miniconda3-py39
conda activate transformer_cuda12
PATH=~/.local/bin:$PATH
echo $PATH
# Run the Python script that uses the GPU
TOKENIZERS_PARALLELISM=false python -u trainer.py
