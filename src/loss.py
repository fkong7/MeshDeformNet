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
import tensorflow as tf
from tensorflow.python.keras import losses
from tensorflow.python.keras import backend as K
import numpy as np

from tensorflow.python.framework import ops
nn_distance_module=tf.load_op_library(os.path.join(os.path.dirname(__file__), '../external/tf_nndistance_so.so'))

def nn_distance(xyz1,xyz2):
    '''
    Computes the distance of nearest neighbors for a pair of point clouds
    input: xyz1: (batch_size,#points_1,3)  the first point cloud
    input: xyz2: (batch_size,#points_2,3)  the second point cloud
    output: dist1: (batch_size,#point_1)   distance from first to second
    output: idx1:  (batch_size,#point_1)   nearest neighbor from first to second
    output: dist2: (batch_size,#point_2)   distance from second to first
    output: idx2:  (batch_size,#point_2)   nearest neighbor from second to first
    '''
    return nn_distance_module.nn_distance(xyz1,xyz2)
@ops.RegisterShape('NnDistance')
def _nn_distance_shape(op):
    shape1=op.inputs[0].get_shape().with_rank(3)
    shape2=op.inputs[1].get_shape().with_rank(3)
    return [tf.TensorShape([shape1.dims[0],shape1.dims[1]]),tf.TensorShape([shape1.dims[0],shape1.dims[1]]),
        tf.TensorShape([shape2.dims[0],shape2.dims[1]]),tf.TensorShape([shape2.dims[0],shape2.dims[1]])]
@ops.RegisterGradient('NnDistance')
def _nn_distance_grad(op,grad_dist1,grad_idx1,grad_dist2,grad_idx2):
    xyz1=op.inputs[0]
    xyz2=op.inputs[1]
    idx1=op.outputs[1]
    idx2=op.outputs[3]
    return nn_distance_module.nn_distance_grad(xyz1,xyz2,grad_dist1,idx1,grad_dist2,idx2)

def laplace_coord(pred, feed_dict, block_id):
    batch_size = tf.shape(pred)[0]
    # Add one zero vertex since the laplace index was initialized to be -1 
    vertex = tf.concat([pred, tf.zeros([batch_size, 1, 3])], 1)
    indices = feed_dict['lape_idx'][block_id-1][:,:-2]
    weights = tf.cast(feed_dict['lape_idx'][block_id-1][:,-1], tf.float32)
    weights = tf.tile(tf.reshape(tf.reciprocal(weights), [-1,1]), [1,3])
    laplace = tf.reduce_sum(tf.gather(vertex, indices, axis=1), 2)
    laplace = tf.subtract(pred, tf.multiply(laplace, weights))
    return laplace

def unit(tensor):
    return tf.nn.l2_normalize(tensor, dim=-1)

def mesh_loss_geometric_cf(feed_dict, block_id, weights, cf_ratio, edge_length):
    def loss(y_true, y_pred):
        losses = mesh_loss(y_pred, y_true, feed_dict, block_id, cf_ratio, edge_length)
        point_loss, edge_loss, normal_loss, laplace_loss = losses
        total_loss = tf.pow(point_loss*100, weights[0])*tf.pow(laplace_loss*100, weights[1]) \
                *tf.pow(normal_loss*100, weights[2])*tf.pow(edge_loss*100, weights[3])
        return total_loss
    return loss

def point_loss_cf(y_true, y_pred):
    gt_pt = y_true[:, :, :3]
    pred_shape = y_pred.get_shape().as_list()
    dist1,idx1,dist2,idx2 = nn_distance(gt_pt, y_pred)
    point_loss = tf.reduce_mean(dist1) + tf.reduce_mean(dist2)
    return point_loss

def laplacian_loss(feed_dict, block_id):
    def k_laplacian_loss(y_true, y_pred):
        num = y_pred.get_shape().as_list()[1] 
        lap = laplace_coord(y_pred, feed_dict, block_id)
        lap_gt = laplace_coord(y_true[:,:num,:3], feed_dict, block_id)
        laplace_loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(lap,lap_gt)), -1))*100.
        return laplace_loss
    return k_laplacian_loss


def edge_loss(feed_dict, block_id):
    def k_edge_loss(y_true, y_pred):
        num = y_pred.get_shape().as_list()[1] 
        gt_pt = y_true[:,:num,:3]
        nod1 = tf.gather(y_pred, feed_dict['edges'][block_id-1][:,0], axis=1)
        nod2 = tf.gather(y_pred, feed_dict['edges'][block_id-1][:,1], axis=1)
        edge = tf.subtract(nod2, nod1)
        nod1_2 = tf.gather(gt_pt, feed_dict['edges'][block_id-1][:,0], axis=1)
        nod2_2 = tf.gather(gt_pt, feed_dict['edges'][block_id-1][:,1], axis=1)
        edge_2 = tf.subtract(nod2_2, nod1_2)
        return tf.reduce_mean(tf.reduce_sum(tf.square(edge_2-edge), axis=-1))
    return k_edge_loss

def normal_loss(feed_dict, block_id):
    def k_normal_loss(y_true, y_pred):
        num = y_pred.get_shape().as_list()[1] 
        gt_pt = y_true[:,:num,:3]
        gt_nm = y_true[:,:num,3:]
        nod1 = tf.gather(y_pred, feed_dict['edges'][block_id-1][:,0], axis=1)
        nod2 = tf.gather(y_pred, feed_dict['edges'][block_id-1][:,1], axis=1)
        edge = tf.subtract(nod2, nod1)
        nod1_2 = tf.gather(gt_pt, feed_dict['edges'][block_id-1][:,0], axis=1)
        nod2_2 = tf.gather(gt_pt, feed_dict['edges'][block_id-1][:,1], axis=1)
        edge_2 = tf.subtract(nod2_2, nod1_2)
        # tf.gather does not support batch_dims in this version
        normal = tf.gather(gt_nm, feed_dict['edges'][block_id-1][:,0], axis=1)
        cosine = tf.reduce_sum(tf.multiply(unit(normal), unit(edge)), axis=-1)
        cosine_2 = tf.reduce_sum(tf.multiply(unit(normal), unit(edge_2)), axis=-1)
        normal_loss = tf.reduce_mean(tf.square(cosine-cosine_2))*1000.
        return normal_loss
    return k_normal_loss


def mesh_loss(pred, gt, feed_dict, block_id, cf_ratio=1., edge_thresh=[0.,0.,0.]):
    gt_pt = gt[:, :, :3] # gt points
    gt_nm = gt[:, :, 3:] # gt normals

    # chafmer distance
    dist1,idx1,dist2,idx2 = nn_distance(gt_pt, pred)
    point_loss = cf_ratio/(cf_ratio+1.)*2*tf.reduce_mean(dist1) + 1./(cf_ratio+1.)*2*tf.reduce_mean(dist2)
    
    # normal cosine loss
    # edge in graph
    nod1 = tf.gather(pred, feed_dict['edges'][block_id-1][:,0], axis=1)
    nod2 = tf.gather(pred, feed_dict['edges'][block_id-1][:,1], axis=1)
    edge = tf.subtract(nod2, nod1)
    edge_length = tf.reduce_sum(tf.square(edge), axis=-1)
    edge_loss = tf.reduce_mean(tf.abs(edge_length-edge_thresh[block_id-1]))
    #edge_loss = tf.reduce_mean(tf.square(edge_length-edge_thresh[block_id-1]))
    ## normal cosine loss
    v1 = tf.gather(pred, feed_dict['faces'][block_id-1][:,0], axis=1)
    v2 = tf.gather(pred, feed_dict['faces'][block_id-1][:,1], axis=1)
    v3 = tf.gather(pred, feed_dict['faces'][block_id-1][:,2], axis=1)
    idx_n = tf.gather(idx2, feed_dict['faces'][block_id-1][:,0], axis=1)
    cross = tf.linalg.cross(v2-v1, v3-v1)

    gt_shape = tf.shape(gt_nm)
    gt_nm = tf.reshape(gt_nm, [gt_shape[0]*gt_shape[1], gt_shape[-1]])
    i_shape = tf.shape(idx_n)
    indices = tf.reshape(idx_n, [-1])
    first = tf.cast(tf.range(tf.size(indices))/i_shape[1], dtype=tf.int32) * gt_shape[1]
    indices = indices + first
    normal= tf.reshape(tf.gather(gt_nm, indices, axis=0), [i_shape[0], i_shape[1], gt_shape[-1]])
    # normal loss weighted by face area
    normal_loss = tf.reduce_mean(tf.reduce_sum(tf.square(unit(normal)-unit(cross)), axis=-1))
    
    lap = laplace_coord(pred, feed_dict, block_id)
    laplace_loss = tf.reduce_mean(tf.reduce_sum(tf.square(lap), -1))
    
    return point_loss, edge_loss, normal_loss, laplace_loss

def dice_coeff(y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score
  

def dice_loss(y_true, y_pred):
    num_class = y_pred.get_shape().as_list()[-1]
    y_true_one_hot = tf.one_hot(tf.cast(y_true,tf.int32), num_class)
    loss = 0.
    for i in range(num_class):
        loss += (1 - dice_coeff_mean(y_true_one_hot[:,:,:,:,:,i], y_pred[:,:,:,:,i]))
    return loss

def dice_coeff_mean(y_true, y_pred):
    smooth = 1.
    # Flatten
    shape = tf.shape(y_pred)
    batch = shape[0]
    length = tf.reduce_prod(shape[1:])
    y_true_f = tf.reshape(y_true, [batch,length])
    y_pred_f = tf.reshape(y_pred, [batch,length])
    intersection = tf.reduce_sum(tf.multiply(y_true_f ,y_pred_f), axis=-1)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f, axis=-1) + tf.reduce_sum(y_pred_f, axis=-1) + smooth)
    return tf.reduce_mean(score)

def bce_dice_loss(y_true, y_pred):
    loss = losses.sparse_categorical_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss
 
def binary_bce_dice_loss(y_true, y_pred):
    condition = tf.greater(y_true, 0)
    res = tf.where(condition, tf.ones_like(y_true), y_true)
    pred = tf.sigmoid(y_pred)
    pred = tf.clip_by_value(pred, 1e-6, 1.-1e-6)
    loss = losses.binary_crossentropy(res, pred) + (1-dice_coeff_mean(res, pred))
    return loss 
    
