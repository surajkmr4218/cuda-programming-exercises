#!/bin/bash
#SBATCH -A csdsx75
#SBATCH -p markov_gpu
#SBATCH --gres=gpu:1
#SBATCH -C l40s
#SBATCH -o batch_size_testing_output_log.o
#SBATCH -e batch_size_testing_error_log.e

module load Python/3.10.4-GCCcore-11.3.0
source venv/bin/activate

nvidia-smi

python batch_size_script.py 
