#!/bin/bash
# Job name:
#SBATCH --job-name=scar
#
# Account:
#SBATCH --account=fc_biome
#
# Partition:
#SBATCH --partition=savio2_htc
#
# QoS:
#SBATCH --qos=savio_normal
#
# Number of nodes:
#SBATCH --nodes=1
#
# Number of tasks (one for each GPU desired for use case) (example):
#SBATCH --ntasks=1
#
# Processors per task (please always specify the total number of processors twice the number of GPUs):
#SBATCH --cpus-per-task=1
#
#Number of GPUs, this can be in the format of "gpu:[1-4]", or "gpu:K80:[1-4] with the type included
##SBATCH --gres=gpu:1
#
# Wall clock limit:
#SBATCH --time=72:00:00
#
## Command(s) to run (example):
module load gcc
module load python/3.6
module load cuda/10.0
module load cudnn/7.5
source ~/tf1/bin/activate  # sh, bash, or zsh
export LD_LIBRARY_PATH=/global/software/sl-7.x86_64/modules/langs/gcc/6.3.0/lib64:$LD_LIBRARY_PATH
#im_dir=/global/scratch/users/fanwei_kong/DeepLearning/ImageData/LAScar/task1/augmentation
im_dir=/global/scratch/users/fanwei_kong/DeepLearning/ImageData/LAScar/task1
python data/data2tfrecords.py --folder ${im_dir} \
    --modality mr \
    --size 128 128 128 \
    --folder_postfix _val \
    --deci_rate 0 \
    --smooth_ite 25 \
    --out_folder ${im_dir}/processed \
    --seg_id 1

deactivate 
