
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
    parser.add_argument('--size', nargs='+', type=int, help='Image dimensions')
    parser.add_argument('--folder_postfix', nargs='?', default='_train', help='Folder postfix of the folder to look for')
    parser.add_argument('--out_folder', help='Name of the output folder')
    parser.add_argument('--deci_rate', type=float, default=0., help='Decimation rate of ground truth mesh')
    parser.add_argument('--smooth_iter', type=int, default=50, help='Smoothing iterations for GT mesh')
    parser.add_argument('--seg_id', default=[], type=int, nargs='+', help='List of segmentation ids to apply marching cube')
    parser.add_argument('--aug_num', type=int, default=0, help='Number of crop augmentation')
    parser.add_argument('--intensity',nargs='+', type=int, default=[750,-750], help='Intensity range to clip to [upper, lower]')
    args = parser.parse_args()
    return args   

def map_polydata_coords(poly, displacement, transform, size):
    coords = vtk_to_numpy(poly.GetPoints().GetData())
    coords += displacement
    coords = np.concatenate((coords,np.ones((coords.shape[0],1))), axis=-1) 
    coords = np.matmul(np.linalg.inv(transform), coords.transpose()).transpose()[:,:3]
    scale = np.array([128, 128, 128])/np.array(size)
    coords *=scale
    return coords

def transform_polydata(poly, displacement, transform, size):
    coords = map_polydata_coords(poly, displacement, transform, size)
    poly.GetPoints().SetData(numpy_to_vtk(coords))
    normals = get_point_normals(poly)
    return np.hstack((coords, normals)), poly

def process_image(image, mask, size, m, intensity, seg_id, deci_rate, smooth_iter):
    img_center = np.array(image.TransformContinuousIndexToPhysicalPoint(np.array(image.GetSize())/2.0))
    
    imgVol = resample_spacing(image, template_size=size, order=1)[0]  # numpy array
    img_center2 = np.array(imgVol.TransformContinuousIndexToPhysicalPoint(np.array(imgVol.GetSize())/2.0))

    spacing = np.array(imgVol.GetSpacing())
    img_py = RescaleIntensity(sitk.GetArrayFromImage(imgVol).transpose(2,1,0), m, intensity).astype(np.float32)
    transform = build_transform_matrix(imgVol)
        
    seg_py = swapLabels_ori(sitk.GetArrayFromImage(mask).transpose(2, 1, 0).astype(np.int64))
    segVol_swap = sitk.GetImageFromArray(seg_py.transpose(2,1,0).astype(np.uint8))
    segVol_swap.CopyInformation(mask)
    mesh_all_list, mesh_all_py_list = [], []
    segVol_swap_vtk = exportSitk2VTK(segVol_swap, spacing=[1.5,1.5,1.5])[0]
    for s, s_id in enumerate(seg_id):
        mesh = smooth_polydata(vtk_marching_cube(segVol_swap_vtk, 0, s_id), iteration=smooth_iter) 
        mesh = bound_polydata_by_image(segVol_swap_vtk, mesh, 1.5)
        mesh = decimation(mesh, deci_rate)
        mesh_py, mesh_poly = transform_polydata(mesh, img_center2-img_center, transform, size)
        mesh_all_list.append(mesh_poly)
        mesh_all_py_list.append(mesh_py)
    # write GT files to disk
    segVol = resample_spacing(mask, template_size=size, order=0)[0]
    seg_py = swapLabels_ori(sitk.GetArrayFromImage(segVol).transpose(2, 1, 0).astype(np.int64))
    return [img_py, mesh_all_py_list, seg_py, transform, spacing], mesh_all_list

def map_scar_to_mesh(scar_mask_py, transform, mesh_coords, scar_id=1):
    # mesh coords are in the physical coordinate system for more accurate mapping
    from scipy.spatial import KDTree
    scar_arr = np.zeros(mesh_coords.shape[0])
    tree = KDTree(mesh_coords)
    
    scar_vox = np.array(np.where(scar_mask_py==scar_id))
    scar_vox_coords = scar_vox.transpose() / 4. # 512//128=4
    distance, indices = tree.query(scar_vox_coords)
    scar_arr[indices] = 1.
    print("Maximum, average and minimum mapping distances are: ", np.max(distance), np.mean(distance), np.min(distance))
    return scar_arr


def process_la_scar_image(image, scar_mask, la_mask, size, m, intensity, seg_id, deci_rate, smooth_iter):
    network_input, mesh_all_list = process_image(image, la_mask, size, m, intensity, seg_id, deci_rate, smooth_iter)
    
    scar_mask = resample_spacing(scar_mask, template_size=[512, 512, 512], order=0)[0] 
    scar_py = swapLabels_ori(sitk.GetArrayFromImage(scar_mask).transpose(2, 1, 0).astype(np.int64))

    ID = 0 # scar for LA only
    mesh_all_list_py = network_input[1]
    mesh = mesh_all_list[ID]
    scar_arr = map_scar_to_mesh(scar_py, build_transform_matrix(scar_mask), vtk_to_numpy(mesh.GetPoints().GetData())) 
    mesh_all_list_py[ID] = np.concatenate((mesh_all_list_py[ID], np.expand_dims(scar_arr, axis=-1)), axis=-1)
    # write image for debugging
    pt_arr = numpy_to_vtk(scar_arr)
    pt_arr.SetName('Scar')
    mesh.GetPointData().AddArray(pt_arr)
    #normed_scar_mask = sitk.GetImageFromArray(scar_py.astype(np.uint8).transpose(2,1,0))
    #normed_scar_mask.SetSpacing([0.25, 0.25, 0.25])
    #normed_im = sitk.GetImageFromArray(network_input[0].transpose(2,1,0))
    #write_vtk_image(exportSitk2VTK(normed_scar_mask)[0], '/Users/fanweikong/Downloads/dataset/task1/train_data/train_1/scar_mask.vti')
    #write_vtk_image(exportSitk2VTK(normed_im)[0], '/Users/fanweikong/Downloads/dataset/task1/train_data/train_1/image.vti')
    return network_input, mesh_all_list

def process_image_w_random_crops(image, mask, size, m, intensity, seg_id, deci_rate, smooth_iter):
    ori_size = np.array(image.GetSize())
    # origin at [0, 1/3], size in [0.75 1]
    crop_ori = (np.random.rand(3)*ori_size/3).astype(np.uint)
    crop_end = (ori_size - np.random.rand(3)*ori_size/4).astype(np.uint)
    cropped = image[crop_ori[0]:crop_end[0], crop_ori[1]:crop_end[1], crop_ori[2]:crop_end[2]]
    tfrecords, mesh_all_list = process_image(cropped, mask, size, m, intensity, seg_id, deci_rate, smooth_iter)
    cropped_mask = mask[crop_ori[0]:crop_end[0], crop_ori[1]:crop_end[1], crop_ori[2]:crop_end[2]]
    segVol = resample_spacing(cropped_mask, template_size=size, order=0)[0]
    tfrecords[2] = swapLabels_ori(sitk.GetArrayFromImage(segVol).transpose(2, 1, 0).astype(np.int64))
    return tfrecords, mesh_all_list, (cropped, mask)

def data_preprocess(modality,data_folder, data_folder_out, fn, intensity, size, seg_id, deci_rate, smooth_iter, aug_num=0):
    for m in modality:
        imgVol_fn , seg_fn = [], []
        for subject_dir in natural_sort(glob.glob(os.path.join(data_folder,m+fn,'*.nii.gz')) \
                +glob.glob(os.path.join(data_folder,m+fn,'*.nii')) ):
            imgVol_fn.append(os.path.realpath(subject_dir))
        for subject_dir in natural_sort(glob.glob(os.path.join(data_folder,m+fn+'_seg','*.nii.gz')) \
                +glob.glob(os.path.join(data_folder,m+fn+'_seg','*.nii')) ):
            seg_fn.append(os.path.realpath(subject_dir))
        print("number of training data %d" % len(imgVol_fn))
        print("number of training data segmentation %d" % len(seg_fn))
        assert len(seg_fn) == len(imgVol_fn)

        mesh_ctr = [np.zeros(3) for i in range(max(1, len(seg_id)))]
        mesh_scale = [0 for i in range(max(1, len(seg_id)))]
        mesh_area = [0 for i in range(max(1, len(seg_id)))]
        num_fns = len(imgVol_fn)
        for i in range(num_fns):
            output_path =  os.path.join(data_folder_out, m+fn, os.path.basename(imgVol_fn[i]))
            img_path, seg_path = imgVol_fn[i], seg_fn[i]
            assert os.path.basename(img_path).split('.')[0] == os.path.basename(seg_path).split('.')[0], "Incosistent image and seg name"
            tfrecords, mesh_all_list = process_image(sitk.ReadImage(img_path), sitk.ReadImage(seg_path), size, m, intensity, seg_id, deci_rate, smooth_iter)
            # compute centroid and radius for mesh initialization 
            for s, (mesh_poly, mesh_py) in enumerate(zip(mesh_all_list, tfrecords[1])):
                mesh_area[s] = mesh_area[s] + get_poly_surface_area(mesh_poly)/num_fns/np.mean(np.array(size))/np.mean(np.array(size))
                mesh_ctr[s] = mesh_ctr[s] + np.mean(mesh_py[:,:3], axis=0)/num_fns/np.array(size)
                mesh_scale[s] += np.mean(np.linalg.norm(mesh_py[:,:3]-np.mean(mesh_py[:,:3], axis=0), axis=-1))/num_fns/np.mean(np.array(size))
            mesh_all = appendPolyData(mesh_all_list)
            #write_vtk_polydata(mesh_all, output_path.split('.nii.gz')[0] +'_ID'+'_'.join(map(str,seg_id))+'.vtp')
            #sitk.WriteImage(sitk.GetImageFromArray(tfrecords[2].astype(np.uint8).transpose(2,1,0)), output_path)
            #sitk.WriteImage(sitk.GetImageFromArray(tfrecords[0].transpose(2,1,0)), output_path.split('.nii.gz')[0]+'_im.nii.gz')
            data_to_tfrecords(*tfrecords, output_path.split('.nii.gz')[0], verbose=True)
            # Random crops - not used in paper
            for aug_n in range(aug_num):
                tfrecords, mesh_all_list, im_mask  = process_image_w_random_crops(sitk.ReadImage(img_path),\
                        sitk.ReadImage(seg_path), size, m, intensity, seg_id, deci_rate, smooth_iter)
                mesh_all = appendPolyData(mesh_all_list)
                data_to_tfrecords(*tfrecords, output_path.split('.nii.gz')[0] + '_{}'.format(aug_n), verbose=True)
        # Save centroid and radius for mesh initialization
        info_fn = os.path.join(data_folder_out, m+fn, 'mesh_info.txt')
        np.savetxt(info_fn, np.hstack((mesh_ctr, np.expand_dims(mesh_scale, -1), np.expand_dims(mesh_area,-1))))

if __name__=='__main__':
    args = parse()
    try:
        os.mkdir(args.out_folder)
    except Exception as e: print(e)
    for m in args.modality:
        try:
            os.mkdir(os.path.join(args.out_folder, m+args.folder_postfix))
        except Exception as e: print(e)
    data_preprocess(args.modality,args.folder, args.out_folder,args.folder_postfix, args.intensity, args.size, args.seg_id, args.deci_rate, args.smooth_iter, args.aug_num)
    # testing
    #image_fn = '/Users/fanweikong/Downloads/dataset/task1/train_data/train_1/enhanced.nii.gz'
    #scar_fn = '/Users/fanweikong/Downloads/dataset/task1/train_data/train_1/scarSegImgM.nii.gz'
    #la_fn = '/Users/fanweikong/Downloads/dataset/task1/train_data/train_1/atriumSegImgMO.nii.gz'
    #m = process_la_scar_image(sitk.ReadImage(image_fn), sitk.ReadImage(scar_fn), sitk.ReadImage(la_fn), [128, 128, 128], \
    #        'mr', [750,-750], [1], 0.2, 25)[-1]
    #write_vtk_polydata(m, '/Users/fanweikong/Downloads/dataset/task1/train_data/train_1/mesh.vtp')

