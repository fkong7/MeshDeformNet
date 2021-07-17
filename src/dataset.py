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
import functools
from augmentation import _augment, _augment_deformnet
def _parse_function(example_proto):
  features = {"X": tf.VarLenFeature(tf.float32),
              "S": tf.VarLenFeature(tf.int64),
              "shape0": tf.FixedLenFeature((), tf.int64),
              "shape1": tf.FixedLenFeature((), tf.int64),
              "shape2": tf.FixedLenFeature((), tf.int64)}
  parsed_features = tf.parse_single_example(example_proto, features)

  img = tf.sparse_tensor_to_dense(parsed_features["X"])
  depth = tf.cast(parsed_features["shape0"], tf.int32)
  height = tf.cast(parsed_features["shape1"], tf.int32)
  width = tf.cast(parsed_features["shape2"], tf.int32)
  label = tf.sparse_tensor_to_dense(parsed_features["S"])
  img = tf.reshape(img, tf.stack([depth,height, width, 1]))
  label = tf.reshape(label, tf.stack([depth, height, width, 1]))
  label = tf.cast(label, tf.int32)
  return img, label

def _parse_function_all(mode):
    def __parse(example_proto):
        if mode=='img':
            features = {"X": tf.VarLenFeature(tf.float32),
              "shape0": tf.FixedLenFeature((), tf.int64),
              "shape1": tf.FixedLenFeature((), tf.int64),
              "shape2": tf.FixedLenFeature((), tf.int64),
              }
            parsed_features = tf.parse_single_example(example_proto, features)
            img = tf.sparse_tensor_to_dense(parsed_features["X"])
            depth = tf.cast(parsed_features["shape0"], tf.int32)
            height = tf.cast(parsed_features["shape1"], tf.int32)
            width = tf.cast(parsed_features["shape2"], tf.int32)
            img = tf.reshape(img, tf.stack([depth,height, width, 1]))
            return img
        elif mode=='seg':
            features = {"S": tf.VarLenFeature(tf.int64),
              "shape0": tf.FixedLenFeature((), tf.int64),
              "shape1": tf.FixedLenFeature((), tf.int64),
              "shape2": tf.FixedLenFeature((), tf.int64),
              }
            parsed_features = tf.parse_single_example(example_proto, features)
            seg = tf.sparse_tensor_to_dense(parsed_features["S"])
            depth = tf.cast(parsed_features["shape0"], tf.int32)
            height = tf.cast(parsed_features["shape1"], tf.int32)
            width = tf.cast(parsed_features["shape2"], tf.int32)
            seg = tf.reshape(seg, tf.stack([depth,height, width, 1]))
            return seg
        elif 'mesh' in mode:
            mesh_id = mode.split('_')[-1]
            features = {"Y_"+mesh_id: tf.VarLenFeature(tf.float32)
              }
            parsed_features = tf.parse_single_example(example_proto, features)
            mesh = tf.sparse_tensor_to_dense(parsed_features["Y_"+mesh_id])

            node_num = tf.cast(tf.shape(mesh)[0]/6, tf.int32)
            mesh = tf.reshape(mesh, tf.stack([node_num, 6 ]))
            return mesh
        elif mode=='transform':
            features = {"Transform": tf.VarLenFeature(tf.float32)}
            parsed_features = tf.parse_single_example(example_proto, features)
            transform = tf.sparse_tensor_to_dense(parsed_features["Transform"])
            transform = tf.reshape(transform, [4, 4])
            return transform
        elif mode=='spacing':
            features = {"Spacing": tf.VarLenFeature(tf.float32)}
            parsed_features = tf.parse_single_example(example_proto, features)
            spacing = tf.sparse_tensor_to_dense(parsed_features["Spacing"])
            spacing = tf.reshape(spacing, [3])
            return spacing
        else:
            raise ValueError('invalid name')

    return __parse

def get_baseline_dataset(filenames, preproc_fn=functools.partial(_augment),
                         threads=5, 
                         batch_size=1,
                         shuffle=True):           
  num_x = len(filenames)
  # Create a dataset from the filenames and labels
  files = tf.data.Dataset.from_tensor_slices(filenames)
  dataset = files.apply(tf.contrib.data.parallel_interleave(
    tf.data.TFRecordDataset, cycle_length=threads))
  # Map our preprocessing function to every element in our dataset, taking
  # advantage of multithreading
  dataset = dataset.map(_parse_function, num_parallel_calls=threads)
  # dataset = dataset.map(_process_pathnames, num_parallel_calls=threads)

  dataset = dataset.map(preproc_fn, num_parallel_calls=threads)
  if shuffle:
    dataset = dataset.shuffle(384)
  
  # It's necessary to repeat our data for all epochs 
  dataset = dataset.repeat().batch(batch_size)
  dataset = dataset.prefetch(buffer_size=batch_size)
  return dataset

def get_baseline_dataset_deformnet(filenames, preproc_fn=functools.partial(_augment_deformnet),
                         threads=1, 
                         batch_size=0,
                         mesh_ids = [2], # default is LV blood pool 2
                         shuffle=True,
                         if_seg=True,
                         shuffle_buffer=10000,
                         num_gcn_blocks=3):           
    num_x = len(filenames)
    # Create a dataset from the filenames and labels
    files = tf.data.Dataset.from_tensor_slices(filenames)
    if shuffle:
        files = files.shuffle(shuffle_buffer)
    dataset = files.apply(tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset, cycle_length=threads))
    # Map our preprocessing function to every element in our dataset, taking
    # advantage of multithreading
    dataset_input = dataset.map(_parse_function_all('img'))
    mesh_list = []
    for i in mesh_ids:
        dataset_mesh = dataset.map(_parse_function_all('mesh_'+str(i)))
        mesh_list.append(dataset_mesh)
    out_list = []
    for i in range(num_gcn_blocks):
        out_list += mesh_list
    if if_seg:
        dataset_seg = dataset.map(_parse_function_all('seg'))
        out_list = [dataset_seg]+out_list
    dataset_output = tf.data.Dataset.zip(tuple(out_list))
    #dataset_output = tf.data.Dataset.zip((dataset_seg, tuple(mesh_list), tuple(mesh_list), tuple(mesh_list)))
    dataset = tf.data.Dataset.zip((dataset_input, dataset_output))
    dataset = dataset.map(preproc_fn)
    dataset = dataset.repeat()
    if batch_size >0:
        dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset
