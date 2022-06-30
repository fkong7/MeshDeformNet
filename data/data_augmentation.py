
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
import sys
import numpy as np
from copy import deepcopy
import argparse
try:
    from mpi4py import MPI
except Exception as e: print(e)

def generate_seg_aug_dataset(im_dir, mask_dir, scar_dir, out_dir, modality, mode='train', AUG_NUM=10,comm=None,rank=0):
    params_affine = {
            'scale_range': [0.75, 1.],
            'rot_range': [-5., 5.],
            'trans_range': [-0., 0.],
            'shear_range': [-0.1, 0.1],
            'flip_prob': 0.
            }
    params_bspline = {
            'num_ctrl_pts': 16,
            'stdev': 4
            }
    sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
    
    import glob
    import SimpleITK as sitk
    from pre_process import AffineTransform, NonlinearTransform, resample_spacing

  
    if rank == 0:
        try:
            os.makedirs(os.path.join(out_dir, modality+'_%s' % mode))
            os.makedirs(os.path.join(out_dir, modality+'_%s_masks' % mode))
            os.makedirs(os.path.join(out_dir, modality+'_%s_scar_masks' % mode))
        except Exception as e: print(e)
        fns = sorted(glob.glob(os.path.join(im_dir, '*.nii.gz'))+
            glob.glob(os.path.join(im_dir, '*.nii')))
        fns_masks = sorted(glob.glob(os.path.join(mask_dir, '*.nii.gz'))+
            glob.glob(os.path.join(mask_dir, '*.nii')))
        fns_scar_masks = sorted(glob.glob(os.path.join(scar_dir, '*.nii.gz'))+
            glob.glob(os.path.join(scar_dir, '*.nii')))
        fns_all, fns_all_masks, fns_all_scar_masks = [], [], []
        nums = []
        for i in range(AUG_NUM):
            fns_all += fns
            fns_all_masks += fns_masks
            fns_all_scar_masks += fns_scar_masks
            nums += [i] * len(fns)

        comm_size = 1 if comm is None else comm.Get_size()
        
        fns_scatter = [None] * comm_size 
        fns_masks_scatter = [None] * comm_size
        fns_scar_masks_scatter = [None] * comm_size
        nums_scatter = [None] * comm_size
        chunck_size = len(fns_all) // comm_size
        
        for i in range(comm_size):
            if i == comm_size-1:
                fns_scatter[i] = fns_all[i*chunck_size:]
                fns_masks_scatter[i] = fns_all_masks[i*chunck_size:]
                fns_scar_masks_scatter[i] = fns_all_scar_masks[i*chunck_size:]
                nums_scatter[i] = nums[i*chunck_size:]
            else:
                fns_scatter[i] = fns_all[i*chunck_size:(i+1)*chunck_size]
                fns_masks_scatter[i] = fns_all_masks[i*chunck_size:(i+1)*chunck_size]
                fns_scar_masks_scatter[i] = fns_all_scar_masks[i*chunck_size:(i+1)*chunck_size]
                nums_scatter[i] = nums[i*chunck_size:(i+1)*chunck_size]
    else:
        nums_scatter, fns_masks_scatter, fns_scar_masks_scatter, fns_scatter = None, None, None, None
        fns_all, fns_all_mask, fns_all_scar_masks, nums = None, None, None, None
    
    if comm is not None:
        nums_scatter = comm.scatter(nums_scatter, root = 0)
        fns_masks_scatter = comm.scatter(fns_masks_scatter, root = 0)
        fns_scar_masks_scatter = comm.scatter(fns_scar_masks_scatter, root = 0)
        fns_scatter  = comm.scatter(fns_scatter, root = 0)

    for fn, fn_mask, fn_scar_mask, aug_id in zip(fns_scatter, fns_masks_scatter, fns_scar_masks_scatter, nums_scatter):
        name = os.path.basename(fn).split(os.extsep, 1)[0]
        name_seg = os.path.basename(fn_mask).split(os.extsep, 1)[0]
        assert name == name_seg , "Image and mask file names do not match!"
        image = sitk.ReadImage(fn)
        mask = sitk.ReadImage(fn_mask)
        scar_mask = sitk.ReadImage(fn_scar_mask)
       
        affine = AffineTransform(image, **params_affine)
        bspline = NonlinearTransform(image, **params_bspline)
        
        bspline.bspline()
        aug_im = bspline.apply_transform(image, order=1)
        aug_mask = bspline.apply_transform(mask, order=0)
        aug_scar_mask = bspline.apply_transform(scar_mask, order=0)
        
        affine.affine()
        aug_im = affine.apply_transform(aug_im, order=1)
        aug_mask = affine.apply_transform(aug_mask, order=0)
        aug_scar_mask = affine.apply_transform(scar_mask, order=0)
        
        affine.clear_transform()
        bspline.clear_transform()
        
        im_out = os.path.join(out_dir, modality+'_%s' % mode, name+'_'+str(aug_id)+'.nii.gz')
        mask_out = os.path.join(out_dir, modality+'_%s_masks' % mode, name+'_'+str(aug_id)+'.nii.gz')
        scar_mask_out = os.path.join(out_dir, modality+'_%s_scar_masks' % mode, name+'_'+str(aug_id)+'.nii.gz')
        
        sitk.WriteImage(aug_im, im_out)
        sitk.WriteImage(aug_scar_mask, scar_mask_out)
        
        mask_aug_py = sitk.GetArrayFromImage(aug_mask)
        for i in np.unique(mask_aug_py):
            tmp = mask_aug_py==i
            true_im = sitk.GetImageFromArray(tmp.astype(np.uint8))
            filt = sitk.BinaryMedianImageFilter()
            filt.SetRadius(3)
            out = filt.Execute(true_im)
            true_im_py = sitk.GetArrayFromImage(out)
            mask_aug_py[true_im_py==1] = i
        mask_aug = sitk.GetImageFromArray(mask_aug_py)
        mask_aug.CopyInformation(aug_im)
        sitk.WriteImage(mask_aug, mask_out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--im_dir', help='Name of the folder with image data')
    parser.add_argument('--seg_dir', help='Name of the folder with la segmentation data')
    parser.add_argument('--scar_dir', help='Name of the folder with scar segmentation data')
    parser.add_argument('--out_dir', help='Name of the output directory')
    parser.add_argument('--modality', help='Modality, ct or mr')
    parser.add_argument('--mode', help='train or val')
    parser.add_argument('--num', type=int, help='Number of augmentations per image')
    args = parser.parse_args()
    
    try:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        total = comm.Get_size()
    except:
        comm = None
        rank = 0

    generate_seg_aug_dataset(args.im_dir, args.seg_dir, args.scar_dir, args.out_dir, args.modality, args.mode,args.num,  comm, rank)
