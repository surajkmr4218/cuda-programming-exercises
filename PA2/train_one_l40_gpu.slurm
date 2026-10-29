#!/bin/bash
#SBATCH -A csdsx75
#SBATCH -p markov_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH -C gpul40s
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH -o q3_l40s_training_output.o
#SBATCH -e q3_l40s_training_error.e

module load Python/3.10.4-GCCcore-11.3.0
source /mnt/vstor/courses/csdsx75/ssk213/cuda-programming-exercises/PA2/venv/bin/activate

echo "Allocated GPU(s): $CUDA_VISIBLE_DEVICES"
nvidia-smi

srun --export=ALL /mnt/vstor/courses/csdsx75/ssk213/cuda-programming-exercises/PA2/venv/bin/python PA2_one_gpu.py