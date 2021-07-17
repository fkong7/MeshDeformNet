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
import tensorflow as tf
import tensorflow.contrib as tfcontrib
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import regularizers
from custom_layers import *
import numpy as np
#from loss import *

class DeformNet(object):
    def __init__(self, batch_size, input_size, feed_dict, amplify_factor=1., num_mesh=1, num_seg=8):
        super(DeformNet, self).__init__()
        self.hidden_dim = 128
        self.num_gcn_block = 3
        self.input_size = input_size
        self.batch_size = batch_size
        self.feed_dict = feed_dict
        self.amplify_factor = amplify_factor
        self.num_mesh = num_mesh
        self.num_seg = num_seg


    def build_keras(self):
        image_inputs = layers.Input(self.input_size, batch_size=self.batch_size)
        features =  self._unet_isensee_encoder(image_inputs) 
        if self.num_seg >0:
            decoder =  self._unet_isensee_decoder(features)
        
        mesh_coords = self.feed_dict['mesh_coords']
        adjs = [j for j in self.feed_dict['adjs']] 
        mesh_coords_p = Position(mesh_coords, self.feed_dict['mesh_scale'],self.feed_dict['mesh_center'])(image_inputs)
        outputs = self._graph_decoder_keras((features, mesh_coords_p, self.feed_dict['mesh_scale']),self.hidden_dim, adjs)
        if self.num_seg >0:
            outputs = [decoder]+ list(outputs)
        return models.Model([image_inputs],outputs)
    
    def _unet_isensee_encoder(self, inputs):
        unet = UNet3DIsensee(self.input_size, num_class=self.num_seg)
        output0 = unet._context_module(16, inputs, strides=(1,1,1))
        output1 = unet._context_module(48, output0, strides=(2,2,2))
        output2 = unet._context_module(96, output1, strides=(2,2,2))
        output3 = unet._context_module(192, output2, strides=(2,2,2))
        output4 = unet._context_module(384, output3, strides=(2,2,2))
        return (output0, output1, output2, output3, output4)

    def _unet_isensee_decoder(self, inputs):
        unet = UNet3DIsensee(self.input_size, num_class=self.num_seg)
        output0, output1, output2, output3, output4 = inputs
        decoder0 = unet._decoder_block(64, [output3, output4])
        decoder1 = unet._decoder_block(32, [output2, decoder0])
        decoder2 = unet._decoder_block(16, [output1, decoder1])
        decoder3 = unet._decoder_block_last_simple(4, [output0, decoder2])
        output0 = layers.Conv3D(unet.num_class, (1, 1, 1))(decoder3)
        output1 = layers.Conv3D(unet.num_class, (1, 1, 1))(decoder2)
        output2_up = layers.UpSampling3D(size=(2,2,2))(layers.Conv3D(unet.num_class, (1, 1, 1))(decoder1))

        output_sum = layers.Add()([output2_up, output1])
        output_sum = layers.UpSampling3D(size=(2,2,2))(output_sum)
        output = layers.Add()([output_sum, output0])
        return output

    def _graph_res_block(self, inputs, adjs, in_dim, hidden_dim):
        output = GraphConv(in_dim ,hidden_dim, act=tf.nn.relu, adjs=adjs)(inputs)
        output2 = GraphConv(in_dim, hidden_dim, act=tf.nn.relu, adjs=adjs)(output)
        return layers.Average()([inputs, output2])

    def _graph_conv_block(self, inputs, adjs, feature_dim, hidden_dim, coord_dim, num_blocks):
        output = GraphConv(feature_dim, hidden_dim, act=tf.nn.relu, adjs=adjs)(inputs)
        output_cat = self._graph_res_block(output, adjs, hidden_dim, hidden_dim)
        for _ in range(num_blocks):
            output_cat = self._graph_res_block(output_cat, adjs, hidden_dim, hidden_dim)
        output = GraphConv(hidden_dim, coord_dim, act=lambda x: x, adjs=adjs)(output_cat)
        return output, output_cat
    
    def _graph_decoder_keras(self, inputs,  hidden_dim, adjs):
       coord_dim = 3
        
       features, mesh_coords, mesh_scale = inputs
       input_size = [float(i) for  i in list(self.input_size)]
       output =  GraphConv(coord_dim, 384, act=tf.nn.relu,adjs=adjs[2])(mesh_coords)
       output = Projection([3,4], input_size)([i for i in features]+[mesh_coords, output])
       output1, output_cat = self._graph_conv_block(output, adjs[2], output.get_shape().as_list()[-1], 288, coord_dim, 3)
       
       # repeat for number of meshes
       output1 = layers.Add()([mesh_coords, ScalarMul(self.amplify_factor/256.)(output1)])
       
       output =  GraphConv(output_cat.get_shape().as_list()[-1], 144, act=tf.nn.relu,adjs=adjs[2])(output_cat)
       output = Projection([1,2], input_size)([i for i in features]+[output1, output])
       output2, output_cat = self._graph_conv_block(output, adjs[2], output.get_shape().as_list()[-1], 96, coord_dim, 3)
       output2 = layers.Add()([output1, ScalarMul(self.amplify_factor/256.)(output2)])
       
       output =  GraphConv(output_cat.get_shape().as_list()[-1], 64, act=tf.nn.relu,adjs=adjs[2])(output_cat)
       output = Projection([0,1], input_size)([i for i in features]+[output2, output])
       output3, _ = self._graph_conv_block(output, adjs[2], output.get_shape().as_list()[-1], 32, coord_dim, 3)
       output3 = layers.Add()([output2, ScalarMul(self.amplify_factor/256.)(output3)])
       
       mesh1 = ScalarMul(128)(output1)
       mesh2 = ScalarMul(128)(output2)
       mesh3 = ScalarMul(128)(output3)
       if self.num_mesh > 1:
           num_coords = mesh_coords.get_shape().as_list()[1] // self.num_mesh
           output1_list = []
           output2_list = []
           output3_list = []
           for i in range(self.num_mesh):
               mesh1_i = layers.Lambda(lambda x: x[:, i*num_coords:(i+1)*num_coords, :])(mesh1)
               mesh2_i = layers.Lambda(lambda x: x[:, i*num_coords:(i+1)*num_coords, :])(mesh2)
               mesh3_i = layers.Lambda(lambda x: x[:, i*num_coords:(i+1)*num_coords, :])(mesh3)
               output1_list.append(mesh1_i)
               output2_list.append(mesh2_i)
               output3_list.append(mesh3_i)
           return output1_list +output2_list + output3_list
       else:
           return [mesh1, mesh2, mesh3]

class UNet3DIsensee(object):
    def __init__(self, input_size, num_class=8, num_filters=[16, 48, 96, 192, 384]):
        super(UNet3DIsensee, self).__init__()
        self.num_class = num_class
        self.input_size = input_size
        self.num_filters = num_filters

    def build(self):
        inputs = layers.Input(self.input_size)

        output0 = self._context_module(self.num_filters[0], inputs, strides=(1,1,1))
        output1 = self._context_module(self.num_filters[1], output0, strides=(2,2,2))
        output2 = self._context_module(self.num_filters[2], output1, strides=(2,2,2))
        output3 = self._context_module(self.num_filters[3], output2, strides=(2,2,2))
        output4 = self._context_module(self.num_filters[4], output3, strides=(2,2,2))
        
        decoder0 = self._decoder_block(self.num_filters[3], [output3, output4])
        decoder1 = self._decoder_block(self.num_filters[2], [output2, decoder0])
        decoder2 = self._decoder_block(self.num_filters[1], [output1, decoder1])
        decoder3 = self._decoder_block_last(self.num_filters[0], [output0, decoder2])
        output0 = layers.Conv3D(self.num_class, (1, 1, 1))(decoder3)
        output1 = layers.Conv3D(self.num_class, (1, 1, 1))(decoder2)
        output2_up = layers.UpSampling3D(size=(2,2,2))(layers.Conv3D(self.num_class, (1, 1, 1))(decoder1))

        output_sum = layers.Add()([output2_up, output1])
        output_sum = layers.UpSampling3D(size=(2,2,2))(output_sum)
        output_sum = layers.Add()([output_sum, output0])
        output = layers.Softmax()(output_sum)

        return models.Model(inputs=[inputs], outputs=[output])

    def _conv_block(self, num_filters, inputs, strides=(1,1,1)):
        output = layers.Conv3D(num_filters, (3, 3, 3),kernel_regularizer=regularizers.l2(0.01),  padding='same', strides=strides)(inputs)
        output = layers.LeakyReLU(alpha=0.01)(InstanceNormalization(axis=-1)(output))
        return output

    def _context_module(self, num_filters, inputs, dropout_rate=0.3, strides=(1,1,1)):
        conv_0 = self._conv_block(num_filters, inputs, strides=strides)
        conv_1 = self._conv_block(num_filters, conv_0)
        dropout = layers.SpatialDropout3D(rate=dropout_rate)(conv_1)
        conv_2 = self._conv_block(num_filters, dropout)
        sum_output = layers.Add()([conv_0, conv_2])
        return sum_output
    
    def _decoder_block(self, num_filters,  inputs, strides=(2,2,2)):
        features, encoder_out = inputs
        upsample = layers.UpSampling3D(size=strides)(encoder_out)
        conv_1 = self._conv_block(num_filters, upsample)
        concat = layers.Concatenate(axis=-1)([conv_1, features])
        conv_2 = self._conv_block(num_filters, concat)
        conv_3 = layers.Conv3D(num_filters, (1,1,1), padding='same')(conv_2)
        output = layers.LeakyReLU(alpha=0.01)(InstanceNormalization(axis=-1)(conv_3))
        return output
    
    def _decoder_block_last_simple(self, num_filters,  inputs, strides=(2,2,2)):
        features, encoder_out = inputs
        upsample = layers.UpSampling3D(size=strides)(encoder_out)
        conv_1 = self._conv_block(num_filters, upsample)
        concat = layers.Concatenate(axis=-1)([conv_1, features])
        conv_2 = self._conv_block(num_filters, concat)
        return conv_2

    def _decoder_block_last(self, num_filters,  inputs, strides=(2,2,2)):
        features, encoder_out = inputs
        upsample = layers.UpSampling3D(size=strides)(encoder_out)
        conv_1 = self._conv_block(num_filters, upsample)
        concat = layers.Concatenate(axis=-1)([conv_1, features])
        conv_2 = self._conv_block(num_filters*2, concat)
        return conv_2
    
