#!/bin/bash
# Job name:
#SBATCH --job-name=scar
#
# Account:
#SBATCH --account=fc_biome
#
# Partition:
#SBATCH --partition=savio2
#
# QoS:
#SBATCH --qos=savio_normal
#
# Number of nodes:
#SBATCH --nodes=2
#
# Number of tasks (one for each GPU desired for use case) (example):
#SBATCH --ntasks-per-node=24
#
# Processors per task (please always specify the total number of processors twice the number of GPUs):
#SBATCH --cpus-per-task=1
#
# Wall clock limit:
#SBATCH --time=72:00:00
#
## Command(s) to run (example):
module load gcc/5.4.0
module load python/3.5
module load boost/1.63.0-gcc
module load hdf5/1.8.18-gcc-p
module load openmpi/2.0.2-gcc
module load netcdf/4.4.1.1-gcc-p
module load cmake/3.7.2
module load swig/3.0.12

im_dir=/global/scratch/users/fanwei_kong/DeepLearning/ImageData/LAScar

mpirun -np 48 python ~/DeformNet/data/data_augmentation.py \
    --im_dir /global/scratch/users/fanwei_kong/DeepLearning/ImageData/LAScar/task1/mr_train \
    --seg_dir /global/scratch/users/fanwei_kong/DeepLearning/ImageData/LAScar/task1/mr_train_masks \
    --scar_dir /global/scratch/users/fanwei_kong/DeepLearning/ImageData/LAScar/task1/mr_train_scar_masks \
    --out_dir /global/scratch/users/fanwei_kong/DeepLearning/ImageData/LAScar/task1/augmentation \
    --modality mr \
    --mode train \
    --num 36
