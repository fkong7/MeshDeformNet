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
out_dir=/global/scratch/users/fanwei_kong/DeepLearning/3DPixel2Mesh
exp_id=LAScar/2022-6-30/scar_all_blocks_wt10
dir=/global/scratch/users/fanwei_kong/DeepLearning/ImageData/LAScar/task1/

python prediction_gcn.py \
    --image $im_dir \
    --mesh_dat data/template_1/data_aux.dat \
    --mesh_txt ${dir}/augmentation/processed/mr_train/mesh_info.txt \
    --attr _val \
    --mesh_tmplt  data/template_1/init3.vtk \
    --model ${out_dir}/output/${exp_id}/weights_gcn.hdf5 \
    --output ${out_dir}/output/${exp_id}/output \
    --modality mr \
    --seg_id 1 \
    --size 128 128 128 \
    --mode test  
