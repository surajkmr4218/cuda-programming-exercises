#!/bin/bash
#SBATCH -A csdsx75
#SBATCH -p markov_gpu
#SBATCH --gres=gpu:1
#SBATCH -C gpu2080
#SBATCH -o output_blas.o
#SBATCH --time=00:30:00
#SBATCH --mem=8G

module load CUDA
nvidia-smi

nvcc -arch=sm_75 -o pa1_blas main_blas.cu -lcublas
stdbuf -oL ./pa1_blas