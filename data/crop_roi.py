
#Copyright (C) 2021 Fanwei Kong, Shawn C. Shadden, University of California, Berkeley

#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
import os
import glob
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../src")) 
import numpy as np
import SimpleITK as sitk
import tensorflow as tf
from utils import *
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from pre_process import swapLabels, RescaleIntensity, resample_spacing, swapLabels_ori
import argparse

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', help='Name of the folder containing the image data')
    parser.add_argument('--modality', nargs='+', help='Name of the modality, mr, ct, split by space')
    parser.add_argument('--folder_postfix', nargs='?', default='_train', help='Folder postfix of the folder to look for')
    parser.add_argument('--out_folder', help='Name of the output folder')
    args = parser.parse_args()
    return args   

def crop_la_scar_image(image, scar_mask, la_mask):
    
    py_seg_scar = sitk.GetArrayFromImage(scar_mask).transpose(2, 1, 0)
    py_la_scar = sitk.GetArrayFromImage(la_mask).transpose(2, 1, 0)
    py_seg = py_seg_scar + py_la_scar
    roi = np.where(py_seg>0)
    roi_x_1, roi_x_2 = max(0, int(np.min(roi[0])-np.random.randint(5, 10))), min(int(np.max(roi[0])+np.random.randint(5, 10)), py_seg.shape[0])
    roi_y_1, roi_y_2 = max(0, int(np.min(roi[1])-np.random.randint(5, 10))), min(int(np.max(roi[1])+np.random.randint(5, 10)), py_seg.shape[1])
    roi_z_1, roi_z_2 = max(0, int(np.min(roi[2])-np.random.randint(5, 10))), min(int(np.max(roi[2])+np.random.randint(5, 10)), py_seg.shape[2])

    print(image.GetSize(), py_seg.shape)
    crop_im = image[roi_x_1:roi_x_2, roi_y_1:roi_y_2, roi_z_1:roi_z_2]
    crop_scar_mask = scar_mask[roi_x_1:roi_x_2, roi_y_1:roi_y_2, roi_z_1:roi_z_2]
    crop_la_mask = la_mask[roi_x_1:roi_x_2, roi_y_1:roi_y_2, roi_z_1:roi_z_2]
    
    return crop_im, crop_scar_mask, crop_la_mask


def crop(modality,data_folder, data_folder_out, fn):
    for m in modality:
        imgVol_fn, scar_fn, seg_fn = [], [], []
        for subject_dir in natural_sort(glob.glob(os.path.join(data_folder,m+fn,'*.nii.gz')) \
                +glob.glob(os.path.join(data_folder,m+fn,'*.nii')) ):
            imgVol_fn.append(os.path.realpath(subject_dir))
        for subject_dir in natural_sort(glob.glob(os.path.join(data_folder,m+fn+'_masks','*.nii.gz')) \
                +glob.glob(os.path.join(data_folder,m+fn+'_masks','*.nii')) ):
            seg_fn.append(os.path.realpath(subject_dir))
        for subject_dir in natural_sort(glob.glob(os.path.join(data_folder,m+fn+'_scar_masks','*.nii.gz')) \
                +glob.glob(os.path.join(data_folder,m+fn+'_scar_masks','*.nii')) ):
            scar_fn.append(os.path.realpath(subject_dir))
        print("number of training data %d" % len(imgVol_fn))
        print("number of training data segmentation %d" % len(seg_fn))
        assert len(seg_fn) == len(imgVol_fn)

        num_fns = len(imgVol_fn)
        for i in range(num_fns):
            output_path =  os.path.join(data_folder_out, m+fn, os.path.basename(imgVol_fn[i]))
            output_path_scar =  os.path.join(data_folder_out, m+fn+'_scar_masks', os.path.basename(imgVol_fn[i]))
            output_path_seg =  os.path.join(data_folder_out, m+fn+'_masks', os.path.basename(imgVol_fn[i]))
            img_path, seg_path, scar_path = imgVol_fn[i], seg_fn[i], scar_fn[i]
            assert os.path.basename(img_path).split('.')[0] == os.path.basename(seg_path).split('.')[0], "Incosistent image and seg name"
            
            crop_im, crop_scar_mask, crop_la_mask = crop_la_scar_image(sitk.ReadImage(img_path), sitk.ReadImage(scar_path), sitk.ReadImage(seg_path))
            
            sitk.WriteImage(crop_im, output_path)
            sitk.WriteImage(crop_scar_mask, output_path_scar)
            sitk.WriteImage(crop_la_mask, output_path_seg)

if __name__=='__main__':
    args = parse()
    try:
        os.mkdir(args.out_folder)
    except Exception as e: print(e)
    for m in args.modality:
        try:
            os.mkdir(os.path.join(args.out_folder, m+args.folder_postfix))
            os.mkdir(os.path.join(args.out_folder, m+args.folder_postfix+'_masks'))
            os.mkdir(os.path.join(args.out_folder, m+args.folder_postfix+'_scar_masks'))
        except Exception as e: print(e)
    
    crop(args.modality,args.folder, args.out_folder,args.folder_postfix)

