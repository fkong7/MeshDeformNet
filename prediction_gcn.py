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
import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
import tensorflow as tf
from tensorflow.python.keras import models as models_keras
import SimpleITK as sitk 
from pre_process import *
from tensorflow.python.keras import backend as K
from model import DeformNet
from data_loader import *
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from utils import *
import argparse
import pickle
import time
from scipy.spatial.distance import directed_hausdorff
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image',  help='Name of the folder containing the image data')
    parser.add_argument('--mesh_dat',  help='Name of the .dat file containing mesh info')
    parser.add_argument('--model',  help='Name of the folder containing the trained model')
    parser.add_argument('--mesh_txt', nargs='+', help='Name of the mesh_info.txt file with tmplt scale and center into')
    parser.add_argument('--mesh_tmplt', help='Name of the finest mesh template')
    parser.add_argument('--attr',  help='Name of the image folder postfix')
    parser.add_argument('--output',  help='Name of the output folder')
    parser.add_argument('--modality', nargs='+', help='Name of the modality, mr, ct, split by space')
    parser.add_argument('--amplify_factor', type=float, default=1., help="amplify_factor of the predicted displacements")
    parser.add_argument('--size', type = int, nargs='+', help='Image dimensions')
    parser.add_argument('--mode', help='Test or validation (without or with ground truth label')
    parser.add_argument('--num_seg', type=int, default=1, help='Number of segmentation classes')
    parser.add_argument('--d_weights', nargs='+', type=float, default=None, help='Weights to down-sample image first')
    parser.add_argument('--ras_spacing',nargs='+', type=float, default=None, help='Prediction spacing')
    parser.add_argument('--seg_id', default=[], type=int, nargs='+', help='List of segmentation ids to apply marching cube')
    args = parser.parse_args()
    return args

import csv
def write_scores(csv_path,scores): 
    with open(csv_path, 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerow(('Dice', 'ASSD'))
        for i in range(len(scores)):
            writer.writerow(tuple(scores[i]))
            print(scores[i])
    writeFile.close()

class Prediction:
    #This class use the GCN model to predict mesh from 3D images
    def __init__(self, info, model_name, mesh_tmplt):
        self.deformnet = DeformNet(**info)
        self.model = self.deformnet.build_keras()
        self.model_name = model_name
        self.model.summary()
        print("Name: ", self.model_name)
        for layer in self.model.layers:
            layer.trainable = False
        self.model.load_weights(self.model_name)
        self.mesh_tmplt = mesh_tmplt
        try:
            os.makedirs(os.path.dirname(self.out_fn))
        except Exception as e: print(e)
         
    def set_image_info(self, modality, image_fn, size, out_fn, mesh_fn=None, d_weights=None, write=False):
        self.modality = modality
        self.image_fn = image_fn
        self.image_vol = load_image_to_nifty(image_fn)
        self.origin = np.array(self.image_vol.GetOrigin())
        self.img_center = np.array(self.image_vol.TransformContinuousIndexToPhysicalPoint(np.array(self.image_vol.GetSize())/2.0))
        self.size = size
        self.out_fn = out_fn
        # down sample to investigate low resolution
        if d_weights:
            self.image_vol = resample_spacing(self.image_vol, template_size = (384, 384, 384), order=1)[0]
            self.image_vol = down_sample_to_slice_thickness(self.image_vol, d_weights, order=0)
            if write:
                dir_name = os.path.dirname(self.out_fn)
                base_name = os.path.basename(self.out_fn)
                sitk.WriteImage(self.image_vol, os.path.join(dir_name, base_name+'_input_downsample.nii.gz'))
        self.image_vol = resample_spacing(self.image_vol, template_size = size, order=1)[0]
        if write:
            sitk.WriteImage(self.image_vol, os.path.join(dir_name, base_name+'_input_linear.nii.gz'))
        self.img_center2 = np.array(self.image_vol.TransformContinuousIndexToPhysicalPoint(np.array(self.image_vol.GetSize())/2.0))
        self.prediction = None
        self.mesh_fn = mesh_fn

    def mesh_prediction(self):
        BLOCK_NUM = self.deformnet.num_gcn_block
        img_vol = sitk.GetArrayFromImage(self.image_vol).transpose(2,1,0)
        img_vol = RescaleIntensity(img_vol,self.modality, [750, -750])
        self.original_shape = img_vol.shape
        transform = build_transform_matrix(self.image_vol)
        spacing = np.array(self.image_vol.GetSpacing())
        model_inputs = [np.expand_dims(np.expand_dims(img_vol, axis=-1), axis=0)]
        start = time.time()
        prediction = self.model.predict(model_inputs)
        end = time.time()
        self.pred_time = end-start
        # remove segmentation
        if self.deformnet.num_seg > 0:
            prediction = prediction[1:]
        num = len(prediction)//BLOCK_NUM
        self.prediction = []
        for i in range(BLOCK_NUM): # block number 
            mesh_list = []
            for k in range(num):
                pred = prediction[i*num+k]
                pred = np.squeeze(pred)
                pred = pred * np.array(self.size)/np.array([128, 128, 128])
                pred = np.concatenate((pred,np.ones((pred.shape[0],1))), axis=-1)  
                pred = np.matmul(transform, pred.transpose()).transpose()[:,:3]
                pred = pred + self.img_center - self.img_center2
                
                new_mesh = vtk.vtkPolyData()
                new_mesh.DeepCopy(self.mesh_tmplt)
                new_mesh.GetPoints().SetData(numpy_to_vtk(pred))
                print("num points: ", new_mesh.GetNumberOfPoints())
                print("num cells: ", new_mesh.GetNumberOfCells())
                mesh_list.append(new_mesh)
            self.prediction.append(mesh_list)
            
    def get_weights(self):
        self.model.load_weights(self.model_name)
        for layer in self.model.layers:
            print(layer.name, layer.get_config())
            weights = layer.get_weights()
            try:
                for w in weights:
                    print(np.max(w), np.min(w))
            except:
                print(weights)
       
    def evaluate_dice(self):
        print("Evaluating dice: ", self.image_fn, self.mesh_fn)
        ref_im = sitk.ReadImage(self.mesh_fn)
        ref_im, M = exportSitk2VTK(ref_im)
        ref_im_py = swapLabels_ori(vtk_to_numpy(ref_im.GetPointData().GetScalars()))
        pred_im_py = vtk_to_numpy(self.seg_result.GetPointData().GetScalars())
        dice_values = dice_score(pred_im_py, ref_im_py)
        return dice_values
    
    def evaluate_assd(self):
        def _get_assd(p_surf, g_surf):
            dist_fltr = vtk.vtkDistancePolyDataFilter()
            dist_fltr.SetInputData(1, p_surf)
            dist_fltr.SetInputData(0, g_surf)
            dist_fltr.SignedDistanceOff()
            dist_fltr.Update()
            distance_poly = vtk_to_numpy(dist_fltr.GetOutput().GetPointData().GetArray(0))
            return np.mean(distance_poly), dist_fltr.GetOutput()
        ref_im =  sitk.ReadImage(self.mesh_fn)
        ref_im = resample_spacing(ref_im, template_size=(256 , 256, 256), order=0)[0]
        ref_im, M = exportSitk2VTK(ref_im)
        ref_im_py = swapLabels_ori(vtk_to_numpy(ref_im.GetPointData().GetScalars()))
        ref_im.GetPointData().SetScalars(numpy_to_vtk(ref_im_py))
        
        dir_name = os.path.dirname(self.out_fn)
        base_name = os.path.basename(self.out_fn)
        pred_im = sitk.ReadImage(os.path.join(dir_name, base_name+'.nii.gz'))
        pred_im = resample_spacing(pred_im, template_size=(256,256,256), order=0)[0]
        pred_im, M = exportSitk2VTK(pred_im)
        pred_im_py = swapLabels_ori(vtk_to_numpy(pred_im.GetPointData().GetScalars()))
        pred_im.GetPointData().SetScalars(numpy_to_vtk(pred_im_py))

        ids = np.unique(ref_im_py)
        pred_poly_l = []
        dist_poly_l = []
        ref_poly_l = []
        dist = [0.]*len(ids)
        #evaluate hausdorff 
        haus = [0.]*len(ids)
        for index, i in enumerate(ids):
            if i==0:
                continue
            p_s = vtk_marching_cube(pred_im, 0, i)
            r_s = vtk_marching_cube(ref_im, 0, i)
            dist_ref2pred, d_ref2pred = _get_assd(p_s, r_s)
            dist_pred2ref, d_pred2ref = _get_assd(r_s, p_s)
            dist[index] = (dist_ref2pred+dist_pred2ref)*0.5

            haus_p2r = directed_hausdorff(vtk_to_numpy(p_s.GetPoints().GetData()), vtk_to_numpy(r_s.GetPoints().GetData()))
            haus_r2p = directed_hausdorff(vtk_to_numpy(r_s.GetPoints().GetData()), vtk_to_numpy(p_s.GetPoints().GetData()))
            haus[index] = max(haus_p2r, haus_r2p)
            pred_poly_l.append(p_s)
            dist_poly_l.append(d_pred2ref)
            ref_poly_l.append(r_s)
        dist_poly = appendPolyData(dist_poly_l)
        pred_poly = appendPolyData(pred_poly_l)
        ref_poly = appendPolyData(ref_poly_l)
        dist_r2p, _ = _get_assd(pred_poly, ref_poly)
        dist_p2r, _ = _get_assd(ref_poly, pred_poly)
        dist[0] = 0.5*(dist_r2p+dist_p2r)

        haus_p2r = directed_hausdorff(vtk_to_numpy(pred_poly.GetPoints().GetData()), vtk_to_numpy(ref_poly.GetPoints().GetData()))
        haus_r2p = directed_hausdorff(vtk_to_numpy(ref_poly.GetPoints().GetData()), vtk_to_numpy(pred_poly.GetPoints().GetData()))
        haus[0] = max(haus_p2r, haus_r2p)

        return dist, haus

    def write_prediction(self, seg_id, ras_spacing=None):
        #fn = '.'.join(self.out_fn.split(os.extsep, -1)[:-1])
        dir_name = os.path.dirname(self.out_fn)
        base_name = os.path.basename(self.out_fn)
        for i, pred in enumerate(self.prediction):
            fn_i =os.path.join(dir_name, 'block'+str(i)+'_'+base_name+'.vtp')
            print("Writing into: ", fn_i)
            pred_all = appendPolyData(pred)
            write_vtk_polydata(pred_all, fn_i)
        _, ext = self.image_fn.split(os.extsep, 1)
        if ext == 'vti':
            ref_im = load_vtk_image(self.image_fn)
        else:
            im = sitk.ReadImage(self.image_fn)
            ref_im, M = exportSitk2VTK(im)
        if ras_spacing is not None:
            ref_im = vtkImageResample(ref_im, ras_spacing, 'NN')    
        out_im_py = np.zeros(vtk_to_numpy(ref_im.GetPointData().GetScalars()).shape)
        for p, s_id in zip(self.prediction[-1], seg_id):
            pred_im = convertPolyDataToImageData(p, ref_im)
            pred_im_py = vtk_to_numpy(pred_im.GetPointData().GetScalars()) 
            if s_id == 7: # hard code for pulmonary artery
                mask = (pred_im_py==1) & (out_im_py==0) 
                out_im_py[mask] = s_id 
            else:
                out_im_py[pred_im_py==1] = s_id
        ref_im.GetPointData().SetScalars(numpy_to_vtk(out_im_py))
        self.seg_result = ref_im
        if ext == 'vti':
            write_vtk_image(ref_im, os.path.join(dir_name, base_name+'.vti'))
        else:
            vtk_write_mask_as_nifty(ref_im, M, self.image_fn, os.path.join(dir_name, base_name+'.nii.gz'))
    def get_score(self):
        return self.score
    def get_score_names(self):
        return self.metric_names
    def get_prediction(self):
        return self.prediction

if __name__ == '__main__':
    args = parse()
    try:
        os.makedirs(args.output)
    except Exception as e: print(e)
    import time
    start = time.time()
    #load image filenames
    BATCH_SIZE = 1
    pkl = pickle.load(open(args.mesh_dat, 'rb'))
    mesh_info = construct_feed_dict(pkl)
    mesh_info['mesh_center'] = [np.zeros(3) for i in range(len(args.seg_id))]
    mesh_info['mesh_scale'] = [0 for i in range(len(args.seg_id))]
    for txt_fn in args.mesh_txt:
        for i in range(len(args.seg_id)):
            ctr_scale = np.loadtxt(txt_fn)
            if len(ctr_scale.shape)==1:
                ctr_scale = np.expand_dims(ctr_scale, axis=0)
            mesh_info['mesh_center'][i] += ctr_scale[i, :-2]/len(args.mesh_txt)
            mesh_info['mesh_scale'][i] += ctr_scale[i, -2]/len(args.mesh_txt)
    mesh_tmplt = load_vtk_mesh(args.mesh_tmplt)
    # write initialization
    init_mesh_l = []
    for i in range(len(args.seg_id)):
        temp = vtk.vtkPolyData()
        temp.DeepCopy(mesh_tmplt)
        coords = vtk_to_numpy(temp.GetPoints().GetData())
        coords = mesh_info['mesh_scale'][i]*coords+mesh_info['mesh_center'][i]
        temp.GetPoints().SetData(numpy_to_vtk(coords))
        init_mesh_l.append(temp)
    init_mesh = appendPolyData(init_mesh_l)
    write_vtk_polydata(init_mesh, os.path.join(args.output, 'init.vtp'))

    print("Mesh center, scale: ", mesh_info['mesh_center'], mesh_info['mesh_scale']) 
    info = {'batch_size': BATCH_SIZE,
            'input_size': (args.size[0], args.size[1], args.size[2], 1),
            'feed_dict': mesh_info,
            'num_mesh': len(args.seg_id),
            'num_seg': args.num_seg,
            'amplify_factor': args.amplify_factor
            }
    filenames = {}
    extensions = ['nii', 'nii.gz', 'vti']
    model_paths = natural_sort(glob.glob(args.model))
    for mdl_id, mdl_fn in enumerate(model_paths):
        predict = Prediction(info, mdl_fn, mesh_tmplt)
        #predict.get_weights()
        for m in args.modality:
            x_filenames, y_filenames = [], []
            for ext in extensions:
                im_loader = DataLoader(m, args.image, fn=args.attr, fn_mask=None if args.mode=='test' else args.attr+'_seg', ext='*.'+ext, ext_out='*.'+ext)
                x_fns_temp, y_fns_temp = im_loader.load_datafiles()
                x_filenames += x_fns_temp
                y_filenames += y_fns_temp
            x_filenames = natural_sort(x_filenames)
            try:
                y_filenames = natural_sort(y_filenames)
            except: pass
            score_list = []
            assd_list = []
            haus_list = []
            time_list = []
            time_list2 = []
            for i in range(len(x_filenames)):
                #set up models
                print("processing "+x_filenames[i])
                start2 = time.time()
                out_fn = os.path.basename(x_filenames[i]).split('.')[0]
                predict.set_image_info(m, x_filenames[i], args.size, os.path.join(args.output, out_fn), y_filenames[i], d_weights=args.d_weights, write=False)
              
                predict.mesh_prediction()
                predict.write_prediction(args.seg_id, args.ras_spacing)
                time_list.append(predict.pred_time)
                end2 = time.time()
                time_list2.append(end2-start2)
                if y_filenames[i] is not None:
                    score_list.append(predict.evaluate_dice())
                    assd, haus = predict.evaluate_assd()
                    assd_list.append(assd)
                    haus_list.append(haus)
            if len(score_list) >0:
                csv_path = os.path.join(args.output, '%s_test.csv' % m)
                csv_path_assd = os.path.join(args.output, '%s_test_assd.csv' % m)
                csv_path_haus = os.path.join(args.output, '%s_test_haus.csv' % m)
                write_scores(csv_path, score_list)
                write_scores(csv_path_assd, assd_list)
                write_scores(csv_path_haus, haus_list)
        del predict 

    end = time.time()
    print("Total time spent: ", end-start)
    print("Avg pred time ", np.mean(time_list)) 
    print("Avg generation time", np.mean(time_list2))
    np.savetxt(os.path.join(args.output, 'avg_pred_time.txt'), np.mean(time_list, keepdims=True))
    np.savetxt(os.path.join(args.output, 'avg_gen_time.txt'), np.mean(time_list2, keepdims=True))
