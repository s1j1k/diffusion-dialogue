#!/bin/bash
#SBATCH --account=punim0811
#SBATCH --partition=feit-gpu-a100
#SBATCH --qos=feit
#SBATCH --gres=gpu:1
#SBATCH --job-name=train-dd
#SBATCH --time=0:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-user=sally.arnold@student.unimelb.edu.au
#SBATCH --mail-type=FAIL,END

module load matplotlib
module load PyTorch/2.1.2-CUDA-11.7.0
python ./src/run_train.py > train_log.txt
