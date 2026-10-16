#!/bin/bash
#SBATCH -A csdsx75
#SBATCH -p markov_gpu
#SBATCH --gres=gpu:1
#SBATCH -o output_log_tc.o
#SBATCH --time=00:30:00
#SBATCH --mem=8G

module load CUDA
nvidia-smi

nvcc -arch=sm_75 -o pa1_tc main_tc.cu
stdbuf -oL ./pa1_tc