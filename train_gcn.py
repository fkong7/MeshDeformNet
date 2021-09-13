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
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
import glob
import functools
import pickle
import argparse
import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.python.keras.optimizers import Adam, SGD
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
from tensorflow.python.keras.utils import multi_gpu_model

from utils import buildImageDataset, construct_feed_dict
from custom_layers import *

from augmentation import change_intensity_img, _augment_deformnet
from dataset import get_baseline_dataset, get_baseline_dataset_deformnet
from model import DeformNet 
from loss import mesh_loss_geometric_cf, point_loss_cf, binary_bce_dice_loss
from call_backs import *
"""# Set up"""

parser = argparse.ArgumentParser()
parser.add_argument('--im_trains',  nargs='+',help='Name of the folder containing the image data')
parser.add_argument('--im_vals', nargs='+', help='Name of the folder containing the image data')
parser.add_argument('--pre_train', default='', help="Filename of the pretrained graph model")
parser.add_argument('--mesh',  help='Name of the .dat file containing mesh info')
parser.add_argument('--mesh_txt', nargs='+', help='Name of the mesh_info.txt file with tmplt scale and center into')
parser.add_argument('--output',  help='Name of the output folder')
parser.add_argument('--attr_trains', nargs='+', help='Attribute name of the folders containing tf records')
parser.add_argument('--attr_vals', nargs='+', help='Attribute name of the folders containing tf records')
parser.add_argument('--train_data_weights', type=float, nargs='+', help='Weights to apply for the samples in different datasets')
parser.add_argument('--val_data_weights', type=float, nargs='+', help='Weights to apply for the samples in different datasets')
parser.add_argument('--file_pattern', default='*.tfrecords', help='Pattern of the .tfrecords files')
parser.add_argument('--modality', nargs='+', help='Name of the modality, mr, ct, split by space')
parser.add_argument('--num_epoch', type=int, help='Maximum number of epochs to run')
parser.add_argument('--num_seg', type=int,default=1, help='Number of segmentation classes')
parser.add_argument('--seg_weight', type=float, default=1., help='Weight of the segmentation loss')
parser.add_argument('--mesh_ids', nargs='+', type=int, default=[2], help='Number of meshes to train')
parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
parser.add_argument('--shuffle_buffer_size', type=int, default=10000, help='Shuffle buffer size')
parser.add_argument('--lr', type=float, help='Learning rate')
parser.add_argument('--cf_ratio', type=float, default=1., help='Loss ratio between gt chamfer loss and pred chamfer loss')
parser.add_argument('--size', type = int, nargs='+', help='Image dimensions')
parser.add_argument('--weights', type = float, nargs='+', help='Loss weights for geometric loss')
parser.add_argument('--hidden_dim', type = int, default=128, help='Hidden dimension')
parser.add_argument('--amplify_factor', type=float, default=1., help="amplify_factor of the predicted displacements")
args = parser.parse_args()

img_shape = (args.size[0], args.size[1], args.size[2], 1)

save_loss_path = args.output
save_model_path = os.path.join(args.output, "weights_gcn.hdf5")

""" Create new directories """
try:
    os.makedirs(os.path.dirname(save_model_path))
    os.makedirs(os.path.dirname(save_loss_path))
except Exception as e: print(e)

"""# Feed in mesh info"""

pkl = pickle.load(open(args.mesh, 'rb'))
mesh_info = construct_feed_dict(pkl)
mesh_info['mesh_center'] = [np.zeros(3) for i in range(len(args.mesh_ids))]
mesh_info['mesh_scale'] = [0 for i in range(len(args.mesh_ids))]
mesh_info['mesh_area'] = [0 for i in range(len(args.mesh_ids))]
mesh_info['edge_length_scaled'] = [np.zeros(3) for i in range(len(args.mesh_ids))] # 3 is number of blocks
for txt_fn in args.mesh_txt:
    for i in range(len(args.mesh_ids)):
        ctr_scale = np.loadtxt(txt_fn) 
        if len(ctr_scale.shape)==1:
            ctr_scale = np.expand_dims(ctr_scale, axis=0) 
        mesh_info['mesh_center'][i] += ctr_scale[i, :-2]/len(args.modality)  
        mesh_info['mesh_scale'][i] += ctr_scale[i, -2]/len(args.modality)  
        mesh_info['mesh_area'][i] += ctr_scale[i, -1]/len(args.modality)

for i in range(len(args.mesh_ids)):
        r = mesh_info['mesh_scale'][i]*2
        scale = r * np.mean(args.size)
        area_ratio = mesh_info['mesh_area'][i]/(4*np.pi*r*r)
        mesh_info['edge_length_scaled'][i] = np.array(mesh_info['edge_length']) * scale * scale * area_ratio
print("Mesh center, scale: ", mesh_info['mesh_center'], mesh_info['mesh_scale'])
print("Mesh edge: ", mesh_info['edge_length_scaled'])

"""## Set up train and validation datasets
Note that we apply image augmentation to our training dataset but not our validation dataset.
"""
tr_cfg = {'change_intensity': {"scale": [0.9, 1.1],"shift": [-0.1, 0.1]}}
tr_preprocessing_fn = functools.partial(_augment_deformnet, **tr_cfg)
if_seg = True if args.num_seg>0 else False

val_preprocessing_fn = functools.partial(_augment_deformnet)
train_ds_list, val_ds_list = [], []
train_ds_num, val_ds_num = [], []
for data_folder_out, attr in zip(args.im_trains, args.attr_trains):
    x_train_filenames_i = buildImageDataset(data_folder_out, args.modality, 41, mode='_train'+attr, ext=args.file_pattern)
    train_ds_num.append(len(x_train_filenames_i))
    train_ds_i = get_baseline_dataset_deformnet(x_train_filenames_i, preproc_fn=tr_preprocessing_fn, mesh_ids=args.mesh_ids, \
            shuffle_buffer=args.shuffle_buffer_size, if_seg=if_seg)
    train_ds_list.append(train_ds_i)
for data_val_folder_out, attr in zip(args.im_vals, args.attr_vals):
    x_val_filenames_i = buildImageDataset(data_val_folder_out, args.modality, 41, mode='_val'+attr, ext=args.file_pattern)
    val_ds_num.append(len(x_val_filenames_i))
    val_ds_i = get_baseline_dataset_deformnet(x_val_filenames_i, preproc_fn=val_preprocessing_fn, mesh_ids=args.mesh_ids, \
            shuffle_buffer=args.shuffle_buffer_size, if_seg=if_seg)
    val_ds_list.append(val_ds_i)
train_data_weights = [w/np.sum(args.train_data_weights) for w in args.train_data_weights]
val_data_weights = [w/np.sum(args.val_data_weights) for w in args.val_data_weights]
print("Sampling probability for train and val datasets: ", train_data_weights, val_data_weights)
train_ds = tf.data.experimental.sample_from_datasets(train_ds_list, weights=train_data_weights)
train_ds = train_ds.batch(args.batch_size)
val_ds = tf.data.experimental.sample_from_datasets(val_ds_list, weights=val_data_weights)
val_ds = val_ds.batch(args.batch_size)

num_train_examples = train_ds_num[np.argmax(train_data_weights)]/np.max(train_data_weights)
num_val_examples =  val_ds_num[np.argmax(val_data_weights)]/np.max(val_data_weights) 
print("Number of train, val samples after reweighting: ", num_train_examples, num_val_examples)

"""# Build the model"""
model = DeformNet(args.batch_size, img_shape, mesh_info, amplify_factor=args.amplify_factor,num_mesh=len(args.mesh_ids), num_seg=args.num_seg)
unet_gcn = model.build_keras()
unet_gcn.summary(line_length=150)

adam = Adam(lr=args.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=True)
output_keys = [node.op.name.split('/')[0] for node in unet_gcn.outputs]
print("Output Keys: ", output_keys)
if args.num_seg >0:
    losses = [ mesh_loss_geometric_cf(mesh_info, 3, args.weights, args.cf_ratio, mesh_info['edge_length_scaled'][(i-1)%len(args.mesh_ids)]) for i in range(1, len(output_keys))]
    losses = [binary_bce_dice_loss] + losses
else:
    losses = [ mesh_loss_geometric_cf(mesh_info, 3, args.weights, args.cf_ratio, mesh_info['edge_length_scaled'][i%len(args.mesh_ids)]) for i in range(len(output_keys))]

losses = dict(zip(output_keys, losses))
metric_loss, metric_key = [], []
for i in range(1, len(args.mesh_ids)+1):
    metric_key.append(output_keys[-i])
    metric_loss.append(point_loss_cf)
metrics_losses = dict(zip(metric_key, metric_loss))
metric_loss_weights = list(np.ones(len(args.mesh_ids)))
loss_weights = list(np.ones(len(output_keys)))
if args.num_seg > 0:
    loss_weights[0] = args.seg_weight


unet_gcn.compile(optimizer=adam, loss=losses,loss_weights=loss_weights,  metrics=metrics_losses)
""" Setup model checkpoint """
save_model_path = os.path.join(args.output, "weights_gcn.hdf5")

cp_cd = SaveModelOnCD(metric_key, save_model_path, patience=50)
lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, min_lr=0.000005)
call_backs = [cp_cd,lr_schedule]

try:
    if args.pre_train != '':
        unet_gcn.load_weights(args.pre_train)
    else:
        unet_gcn.load_weights(save_model_path)
except Exception as e:
  print("Model not loaded", e)

""" Training """
history =unet_gcn.fit(train_ds, 
                   steps_per_epoch=int(np.ceil(500./float(args.batch_size))),
                   epochs=args.num_epoch,
                   validation_data=val_ds,
                   validation_steps= int(np.ceil(num_val_examples / float(args.batch_size))),
                   callbacks=call_backs)
with open(save_loss_path+"_history", 'wb') as handle: # saving the history 
        pickle.dump(history.history, handle)

