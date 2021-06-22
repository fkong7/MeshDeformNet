# DeformNet

This repository contains the source code for our paper:

Kong, F., Wilson, N., Shadden, S.C.: A deep-learning approach for direct whole-heart mesh reconstruction (2021)

We propose a novel deep-learning-based approach that directly predicts whole heart surface meshes from volumetric CT and MR image data.

<img width="800" alt="network3" src="https://user-images.githubusercontent.com/31931939/122881479-10c10000-d2f0-11eb-8d52-ecbc615d7817.png">

Our method demonstrated promising performance of generating high-resolution and high-quality whole heart reconstructions. We are able to generate temporally-consistent and feature-corresponding surface mesh predictions for heart motion from CT or MR cine sequences.

<img width="550" alt="examples" src="https://user-images.githubusercontent.com/31931939/122882003-993fa080-d2f0-11eb-8599-4d476b082f18.png"> <img width="260" alt="4dct2-_1_" src="https://user-images.githubusercontent.com/31931939/122882976-93968a80-d2f1-11eb-99b4-41a30a2ca2ee.gif">

## Dependencies

- Tensorflow V 1.14
- SimpleITK 
- VTK

## Data Augmentation

Data augmentation were applied on the training data. Specifically, we applied random scaling, random rotation, random shearing as well as elastic deformations. The augmemtation script can run in parallel using MPI.

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

Comming soon...

## Prediction

Comming soon...
