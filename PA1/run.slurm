#!/bin/bash
#SBATCH -A csdsx75
#SBATCH -p markov_gpu
#SBATCH --gres=gpu:1
#SBATCH -o output_log.o
#SBATCH --time=02:00:00
#SBATCH --mem=8G

module load CUDA
nvidia-smi

nvcc -O3 -arch=sm_75 -o pa1 main.cu
./pa1