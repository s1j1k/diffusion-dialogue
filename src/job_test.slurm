#!/bin/bash
#SBATCH --account=punim0811
#SBATCH --partition=gpu-a100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --job-name=cifar10
#SBATCH --time=0:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-user=sjarnold@student.unimelb.edu.au
#SBATCH --mail-type=FAIL,END
module load PyTorch/2.1.2-CUDA-11.7.0
module load torchvision/0.13.1-CUDA-11.7.0
python run_decode.py > test_log.txt
