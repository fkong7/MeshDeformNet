#!/bin/bash
# Job name:
#SBATCH --job-name=scar100
#
# Account:
#SBATCH --account=fc_biome
#
# Partition:
##SBATCH --partition=savio2_1080ti
#SBATCH --partition=savio3_gpu
#
# QoS:
##SBATCH --qos=savio_normal
#SBATCH --qos=gtx2080_gpu3_normal
#
# Number of nodes:
#SBATCH --nodes=1
#
# Number of tasks (one for each GPU desired for use case) (example):
#SBATCH --ntasks=1
#
# Processors per task (please always specify the total number of processors twice the number of GPUs):
#SBATCH --cpus-per-task=2
#
#Number of GPUs, this can be in the format of "gpu:[1-4]", or "gpu:K80:[1-4] with the type included
##SBATCH --gres=gpu:1
#SBATCH --gres=gpu:GTX2080TI:1
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
export HDF5_USE_FILE_LOCKING='FALSE'
out_dir=/global/scratch/users/fanwei_kong/DeepLearning/3DPixel2Mesh
exp_id=LAScar/2022-7-17/scar_all_blocks_focalAlpha075pow07
dir=/global/scratch/users/fanwei_kong/DeepLearning/ImageData/LAScar/task1/
python train_gcn.py \
    --im_train ${dir}/augmentation/processed \
    --im_val ${dir}/augmentation/processed \
    --mesh_txt ${dir}/augmentation/processed/mr_train/mesh_info.txt \
    --mesh data/template_1/data_aux.dat \
    --attr_trains '' \
    --attr_vals '' \
    --train_data_weights 1. \
    --val_data_weights 1. \
    --output ${out_dir}/output/${exp_id}\
    --modality mr \
    --num_epoch 500 \
    --batch_size 1 \
    --lr 0.001 \
    --size 128 128 128 \
    --weights 0.29336324 0.05 0.46902252 0.16859047 0.7 \
    --mesh_ids 0 \
    --seg_weight 100
    #--attr_trains _single \
    #--attr_vals _single \
    #--attr_trains '' \
    #--attr_vals '' \
    #--pre_train ${out_dir}/output/LAScar/2022-6-30/scar_all_blocks_wt0/weights_gcn.hdf5 \

#python prediction_gcn.py \
#    --image /global/scratch-old/fanwei_kong/DeepLearning/3DPixel2Mesh/data/multidataset_whole_heart \
#    --mesh_dat data/template/data_aux.dat \
#    --mesh_txt data/template/mesh_info_ct.txt data/template/mesh_info_mr.txt\
#    --attr _val \
#    --mesh_tmplt data/template/init3.vtk \
#    --model ${out_dir}/output/2021-9-7/public_validate/${exp_id}/weights_gcn.hdf5 \
#    --output ${out_dir}/output/2021-9-7/public_validate/${exp_id}/val \
#    --modality ct mr \
#    --seg_id 1 2 3 4 5 6 7\
#    --size 128 128 128 \
#    --mode test
deactivate 
