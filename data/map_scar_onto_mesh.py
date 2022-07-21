
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
import argparse



def map_scar_to_mesh(scar_mask, la_mask, smooth_iter=25, deci_rate=0.):
    # mesh coords are in the physical coordinate system for more accurate mapping
    from scipy.spatial import KDTree
    print(la_mask) 
    la_mask_vtk =exportSitk2VTK(la_mask, spacing=[1.,1.,1.])[0]
    la_id = np.unique(sitk.GetArrayFromImage(la_mask).transpose(2, 1, 0))[-1]
    print("la_id: ", la_id)
    la_mesh = smooth_polydata(vtk_marching_cube(la_mask_vtk, 0, la_id), iteration=smooth_iter)  
    la_mesh = decimation(la_mesh, deci_rate)

    mesh_coords = vtk_to_numpy(la_mesh.GetPoints().GetData())

    scar_arr = np.zeros(mesh_coords.shape[0])
    tree = KDTree(mesh_coords)


    size = [256, 256, 256]
    new_spacing = np.array(scar_mask.GetSpacing())*np.array(scar_mask.GetSize())/np.array(size)
    scar_mask = sitk.Resample(scar_mask, size,
                         sitk.Transform(),
                         sitk.sitkNearestNeighbor,
                         scar_mask.GetOrigin(),
                         new_spacing,
                         scar_mask.GetDirection(),
                         0,
                         scar_mask.GetPixelID())
    transform = build_transform_matrix(scar_mask)
    scar_mask_py = sitk.GetArrayFromImage(scar_mask).transpose(2, 1, 0)
    scar_id = np.unique(scar_mask_py)[-1]
    scar_vox = np.array(np.where(scar_mask_py==scar_id)).transpose()
    scar_vox = np.concatenate((scar_vox, np.ones((scar_vox.shape[0],1))), axis=-1)
    scar_vox = np.matmul(transform, scar_vox.transpose()).transpose()[:,:3]

    distance, indices = tree.query(scar_vox)
    scar_arr[indices] = 1.
    print("Maximum, average and minimum mapping distances are: ", np.max(distance), np.mean(distance), np.min(distance))

    scar_arr_vtk = numpy_to_vtk(scar_arr)
    scar_arr_vtk.SetName('Scar')
    la_mesh.GetPointData().AddArray(scar_arr_vtk)
    return la_mesh



def map_scar(modality,data_folder, data_folder_out, fn, deci_rate, smooth_iter):
    for m in modality:
        scar_fn, seg_fn = [], []
        for subject_dir in natural_sort(glob.glob(os.path.join(data_folder,m+fn+'_masks','*.nii.gz')) \
                +glob.glob(os.path.join(data_folder,m+fn+'_masks','*.nii')) ):
            seg_fn.append(os.path.realpath(subject_dir))
        for subject_dir in natural_sort(glob.glob(os.path.join(data_folder,m+fn+'_scar_masks','*.nii.gz')) \
                +glob.glob(os.path.join(data_folder,m+fn+'_scar_masks','*.nii')) ):
            scar_fn.append(os.path.realpath(subject_dir))
        assert len(seg_fn) == len(scar_fn)

        num_fns = len(seg_fn)
        for i in range(num_fns):
            output_path =  os.path.join(data_folder_out, m+fn, os.path.basename(seg_fn[i]))
            seg_path, scar_path = seg_fn[i], scar_fn[i]
            la_mesh = map_scar_to_mesh(sitk.ReadImage(scar_path), sitk.ReadImage(seg_path), smooth_iter, deci_rate)

            write_vtk_polydata(la_mesh, output_path+'.vtp')
            

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', help='Name of the folder containing the image data')
    parser.add_argument('--modality', nargs='+', help='Name of the modality, mr, ct, split by space')
    parser.add_argument('--folder_postfix', nargs='?', default='_train', help='Folder postfix of the folder to look for')
    parser.add_argument('--out_folder', help='Name of the output folder')
    parser.add_argument('--deci_rate', type=float, default=0., help='Decimation rate of ground truth mesh')
    parser.add_argument('--smooth_iter', type=int, default=50, help='Smoothing iterations for GT mesh')
    args = parser.parse_args()
    try:
        os.mkdir(args.out_folder)
    except Exception as e: print(e)
    for m in args.modality:
        try:
            os.mkdir(os.path.join(args.out_folder, m+args.folder_postfix))
        except Exception as e: print(e)
    map_scar(args.modality,args.folder, args.out_folder,args.folder_postfix, args.deci_rate, args.smooth_iter)
    # testing
    '''
    python map_scar_onto_mesh.py  --folder /Users/fanweikong/Documents/Modeling/MeshDeformNet/LAScar/cropped \
            --modality mr \
            --folder_postfix _train_debug \
            --out_folder /Users/fanweikong/Documents/Modeling/MeshDeformNet/LAScar/cropped
    '''      
