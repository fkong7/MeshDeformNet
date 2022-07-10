# MeshDeformNet

This repository contains the source code for our paper:

Kong, F., Wilson, N., Shadden, S.C.: A deep-learning approach for direct whole-heart mesh reconstruction (2021)

We propose a novel deep-learning-based approach that directly predicts whole heart surface meshes from volumetric CT and MR image data.

<img width="800" alt="network3" src="https://user-images.githubusercontent.com/31931939/122881479-10c10000-d2f0-11eb-8d52-ecbc615d7817.png">

Our method demonstrated promising performance of generating high-resolution and high-quality whole heart reconstructions. We are able to generate temporally-consistent and feature-corresponding surface mesh predictions for heart motion from CT or MR cine sequences.

<img width="550" alt="examples" src="https://user-images.githubusercontent.com/31931939/122882003-993fa080-d2f0-11eb-8599-4d476b082f18.png"> <img width="260" alt="4dct2-_1_" src="https://user-images.githubusercontent.com/31931939/122882976-93968a80-d2f1-11eb-99b4-41a30a2ca2ee.gif">

## Data Augmentation

Data augmentation were applied on the training data. Specifically, we applied random scaling, random rotation, random shearing as well as elastic deformations. The augmemtation script can run in parallel using MPI. For Section3.1-3.5, we used `--num 10` and `--num 20` for CT and MR data, respectively. For Section 3.6, we used `--num 60` for both CT and MR images.

```
mpirun -n 20 python data/data_augmentation.py.py \
    --im_dir  /path/to/image/data \
    --seg_dir  /path/to/segmentation/data \
    --out_dir  /path/to/output \
    --modality ct \ # ct or mr
    --mode val \ # train or val
    --num 10 # number of augmentated copies per image
```

## Data Pre-Processing

The data pre-processing script will apply intensity normalization and resize the image data. The pre-processed images and segmentation will be converted to .tfrecords.

```
python data/data2tfrecords.py --folder /path/to/top/image/directory \
    --modality ct mr \
    --size 128 128 128 \ # image dimension for training
    --folder_postfix _train \ # _train or _val, i.e. will process the images/segmentation in ct_train and ct_train_seg
    --deci_rate 0  \ # decimation rate on ground truth surface meshes
    --smooth_ite 50 \ # Laplacian smoothing on ground truth surface meshes
    --out_folder /path/to/output \
    --seg_id 1 2 3 4 5 6 7 # segmentation ids, 1-7 for seven cardiac structures
```

## Training

The template mesh and the relavant mesh information (e.g. Laplacian matrix) required for training/testing are stored in data/template/init3.vtk and data/template/data_aux.dat. To use a different template sphere mesh or to use different number of spheres, you woule need to run `make_auxiliary_dat_file.py`. For example:

```
python make_auxiliary_dat_file.py --template_fn data/template/init3.vtk --num_meshes 7 --output /path/to/output
```

You would need to use the following script to start training. Our pre-trained weights are in `weights/weights_gcn.hdf5`.

```
python train_gcn.py \
    --im_train /path/to/top/tfrecords/directory \
    --im_val /path/to/top/tfrecords/directory \
    --mesh_txt  data/template/mesh_info_ct.txt data/template/mesh_info_mr.txt \ # template mesh initialization
    --mesh data/template/data_aux.dat \ # template mesh data
    --attr_trains '' \ # specify name of different validation dataset (e.g. '' _aug for ct_train, ct_train_aug)
    --attr_vals '' \ # specify names of different validation dataset (e.g. '' _aug for ct_val, ct_val_aug)
    --train_data_weights 1. \ # weights for different training dataset
    --val_data_weights 1. \ # weights for different validation dataset
    --output /path/to/output \
    --modality ct mr \
    --num_epoch 500 \
    --batch_size 1 \
    --lr 0.001 \
    --size 128 128 128 \ # input dimension
    --weights 0.29336324 0.05 0.46902252 0.16859047 \ # weights for point, laplacian, normal, and edge losses 
    --mesh_ids 0 1 2 3 4 5 6 \ # ids of cardiac structures, starts at 0
    --num_seg 1 \ # number of class for segmentation output, 1 for binary segmentation, 0 to turn off the segmentation module.
    --seg_weight 100. # weight on segmentation loss
```

## Prediction

You would need to use the following script to generate predictions.

```
python prediction_gcn.py \
    --image /path/to/top/image/directory \
    --mesh_dat data/template/data_aux.dat \
    --mesh_txt data/template/mesh_info_ct.txt data/template/mesh_info_mr.txt \
    --attr _test \ # name of folder where image data are stored: (i.e. /path/to/top/image/directory/ct_test)
    --mesh_tmplt  data/template/init3.vtk \
    --model weights/weights_gcn.hdf5 \
    --output /path/to/output \
    --modality ct mr \
    --seg_id 1 2 3 4 5 6 7\ # segmentation ids, 1-7 for seven cardiac structures
    --size 128 128 128 \ # input dimension
    --mode test  # test - generate predictions only; validate - also compute accuracy metrics
```
