#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuA100 
#SBATCH --time=8:00:00
#SBATCH --job-name=factcheck_sweep_agent
#SBATCH --output=sweep-%j.out
 
# Activate environment
uenv verbose cuda-12.2.0 cudnn-12.x-8.8.0
uenv miniconda3-py39
conda activate transformer_cuda12
PATH=~/.local/bin:$PATH
echo $PATH

# Ensure wandb is installed and login if needed
pip install wandb --upgrade
# Replace YOUR_API_KEY with your actual wandb API key
wandb login YOUR_API_KEY

# Launch a wandb agent for a specific sweep
# Replace 'your_sweep_id' with your actual wandb sweep ID
wandb agent thwjm4wb