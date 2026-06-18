#!/bin/bash
#SBATCH --partition=mlgpu_medium
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH -N 1
#SBATCH --gres=gpu:1                


#SBATCH -o slurm.%N.%j.out
#SBATCH -e slurm.%N.%j.err








source /home/s59efara_hpc/cpcl_cov/setup.sh



python /home/s59efara_hpc/covariance/cpcl_gpu.py