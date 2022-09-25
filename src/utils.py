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
import glob
import re
import vtk
try:
    import tensorflow as tf
    from tensorflow.python.keras import backend as K
except Exception as e: print(e)

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def dice_score(pred, true):
    pred = pred.astype(np.int)
    true = true.astype(np.int)
    num_class = np.unique(true)

    #change to one hot
    dice_out = [None]*len(num_class)
    for i in range(1, len(num_class)):
        pred_c = pred == num_class[i]
        true_c = true == num_class[i]
        dice_out[i] = np.sum(pred_c*true_c)*2.0 / (np.sum(pred_c) + np.sum(true_c))

    mask =( pred > 0 )+ (true > 0)
    dice_out[0] = np.sum((pred==true)[mask]) * 2. / (np.sum(pred>0) + np.sum(true>0))
    return dice_out

def smooth_polydata(poly, iteration=25, boundary=False, feature=False):
    """
    This function smooths a vtk polydata
    Args:
        poly: vtk polydata to smooth
        boundary: boundary smooth bool
    Returns:
        smoothed: smoothed vtk polydata
    """

    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(poly)
    smoother.SetBoundarySmoothing(boundary)
    smoother.SetFeatureEdgeSmoothing(feature)
    smoother.SetNumberOfIterations(iteration)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()

    smoothed = smoother.GetOutput()

    return smoothed

def bound_polydata_by_image(image, poly, threshold):
    import vtk
    bound = vtk.vtkBox()
    image.ComputeBounds()
    b_bound = image.GetBounds()
    b_bound = [b+threshold if (i % 2) ==0 else b-threshold for i, b in enumerate(b_bound)]
    print("Bounding box: ", b_bound)
    bound.SetBounds(b_bound)
    clipper = vtk.vtkClipPolyData()
    clipper.SetClipFunction(bound)
    clipper.SetInputData(poly)
    clipper.InsideOutOn()
    clipper.Update()
    return clipper.GetOutput()

def decimation(poly, rate):
    import vtk
    """
    Simplifies a VTK PolyData
    Args: 
        poly: vtk PolyData
        rate: target rate reduction
    """
    decimate = vtk.vtkQuadricDecimation()
    decimate.SetInputData(poly)
    decimate.AttributeErrorMetricOn()
    decimate.SetTargetReduction(rate)
    decimate.VolumePreservationOn()
    decimate.Update()
    return decimate.GetOutput()

def get_largest_connected_polydata(poly):
    import vtk
    connectivity = vtk.vtkPolyDataConnectivityFilter()
    connectivity.SetInputData(poly)
    connectivity.SetExtractionModeToLargestRegion()
    connectivity.Update()
    poly = connectivity.GetOutput()
    return poly

def get_poly_surface_area(poly):
    import vtk
    mass = vtk.vtkMassProperties()
    mass.SetInputData(poly)
    mass.Update()
    return mass.GetSurfaceArea()

def vtkImageResample(image, spacing, opt):
    """
    Resamples the vtk image to the given dimenstion
    Args:
        image: vtk Image data
        spacing: image new spacing
        opt: interpolation option: linear, NN, cubic
    Returns:
        image: resampled vtk image data
    """
    import vtk
    reslicer = vtk.vtkImageReslice()
    reslicer.SetInputData(image)
    if opt=='linear':
        reslicer.SetInterpolationModeToLinear()
    elif opt=='NN':
        reslicer.SetInterpolationModeToNearestNeighbor()
    elif opt=='cubic':
        reslicer.SetInterpolationModeToCubic()
    else:
        raise ValueError("interpolation option not recognized")

    #size = np.array(image.GetSpacing())*np.array(image.GetDimensions())
    #new_spacing = size/np.array(dims)

    reslicer.SetOutputSpacing(*spacing)
    reslicer.Update()

    return reslicer.GetOutput()

def convertPolyDataToImageData(poly, ref_im):
    """
    Convert the vtk polydata to imagedata 
    Args:
        poly: vtkPolyData
        ref_im: reference vtkImage to match the polydata with
    Returns:
        output: resulted vtkImageData
    """
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
    
    ref_im.GetPointData().SetScalars(numpy_to_vtk(np.zeros(vtk_to_numpy(ref_im.GetPointData().GetScalars()).shape)))
    ply2im = vtk.vtkPolyDataToImageStencil()
    ply2im.SetTolerance(0.05)
    ply2im.SetInputData(poly)
    ply2im.SetOutputSpacing(ref_im.GetSpacing()) 
    ply2im.SetInformationInput(ref_im)
    ply2im.Update()

    stencil = vtk.vtkImageStencil()
    stencil.SetInputData(ref_im)
    stencil.ReverseStencilOn()
    stencil.SetStencilData(ply2im.GetOutput())
    stencil.Update()
    output = stencil.GetOutput()
 
    return output

def exportSitk2VTK(sitkIm,spacing=None):
    """
    This function creates a vtk image from a simple itk image
    Args:
        sitkIm: simple itk image
    Returns:
        imageData: vtk image
import SimpleITK as sitk
    """
    if not spacing:
        spacing = sitkIm.GetSpacing()
    import SimpleITK as sitk
    import vtk
    img = sitk.GetArrayFromImage(sitkIm).transpose(2,1,0)
    vtkArray = exportPython2VTK(img)
    imageData = vtk.vtkImageData()
    imageData.SetDimensions(sitkIm.GetSize())
    imageData.GetPointData().SetScalars(vtkArray)
    imageData.SetOrigin([0.,0.,0.])
    imageData.SetSpacing(spacing)
    matrix = build_transform_matrix(sitkIm)
    space_matrix = np.diag(list(spacing)+[1.])
    matrix = np.matmul(matrix, np.linalg.inv(space_matrix))
    matrix = np.linalg.inv(matrix)
    vtkmatrix = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            vtkmatrix.SetElement(i, j, matrix[i,j])
    reslice = vtk.vtkImageReslice()
    reslice.SetInputData(imageData)
    reslice.SetResliceAxes(vtkmatrix)
    reslice.SetInterpolationModeToNearestNeighbor()
    reslice.Update()
    imageData = reslice.GetOutput()
    #imageData.SetDirectionMatrix(sitkIm.GetDirection())

    return imageData, vtkmatrix

def load_vtk_image(fn, mode='linear'):
    """
    This function imports image file as vtk image.
    Args:
        fn: filename of the image data
    Return:
        label: label map as a vtk image
    """
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
    name_list = fn.split(os.extsep)
    ext = name_list[-1]

    if ext=='vti':
        reader = vtk.vtkXMLImageDataReader()
        reader.SetFileName(fn)
        reader.Update()
        label = reader.GetOutput()
    elif ext=='nii' or '.'.join([name_list[-2], ext])=='nii.gz':
        reader = vtk.vtkNIFTIImageReader()
        reader.SetFileName(fn)
        reader.Update()
        image = reader.GetOutput()
        matrix = reader.GetQFormMatrix()
        if matrix is None:
            matrix = reader.GetSFormMatrix()
        matrix.Invert()
        Sign = vtk.vtkMatrix4x4()
        Sign.Identity()
        Sign.SetElement(0, 0, -1)
        Sign.SetElement(1, 1, -1)
        M = vtk.vtkMatrix4x4()
        M.Multiply4x4(matrix, Sign, M)
        reslice = vtk.vtkImageReslice()
        reslice.SetInputData(image)
        reslice.SetResliceAxes(M)
        if mode=='linear':
            reslice.SetInterpolationModeToLinear()
        else:
            reslice.SetInterpolationModeToNearestNeighbor()
        reslice.SetOutputSpacing(np.min(image.GetSpacing())*np.ones(3))
        reslice.Update()
        label = reslice.GetOutput()
        py_label = vtk_to_numpy(label.GetPointData().GetScalars())
        py_label = (py_label + reader.GetRescaleIntercept())/reader.GetRescaleSlope()
        label.GetPointData().SetScalars(numpy_to_vtk(py_label))
    else:
        raise IOError("File extension is not recognized: ", ext)
    return label

def vtk_write_mask_as_nifty2(mask, image_fn, mask_fn):
    import vtk
    origin = mask.GetOrigin()
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(image_fn)
    reader.Update()
    writer = vtk.vtkNIFTIImageWriter()
    Sign = vtk.vtkMatrix4x4()
    Sign.Identity()
    Sign.SetElement(0, 0, -1)
    Sign.SetElement(1, 1, -1)
    M = reader.GetQFormMatrix()
    if M is None:
        M = reader.GetSFormMatrix()
    M2 = vtk.vtkMatrix4x4()
    M2.Multiply4x4(Sign, M, M2)
    reslice = vtk.vtkImageReslice()
    reslice.SetInputData(mask)
    reslice.SetResliceAxes(M2)
    reslice.SetInterpolationModeToNearestNeighbor()
    reslice.Update()
    mask = reslice.GetOutput()
    mask.SetOrigin([0.,0.,0.])

    writer.SetInputData(mask)
    writer.SetFileName(mask_fn)
    writer.SetQFac(reader.GetQFac())
    q_mat = reader.GetQFormMatrix()
    writer.SetQFormMatrix(q_mat)
    s_mat = reader.GetSFormMatrix()
    writer.SetSFormMatrix(s_mat)
    writer.Write()
    return

def vtk_write_mask_as_nifty(mask,M , image_fn, mask_fn):
    import vtk
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(image_fn)
    reader.Update()
    writer = vtk.vtkNIFTIImageWriter()
    M.Invert()
    if reader.GetQFac() == -1:
        for i in range(3):
            temp = M.GetElement(i, 2)
            M.SetElement(i, 2, temp*-1)
    reslice = vtk.vtkImageReslice()
    reslice.SetInputData(mask)
    reslice.SetResliceAxes(M)
    reslice.SetInterpolationModeToNearestNeighbor()
    reslice.Update()
    mask = reslice.GetOutput()
    mask.SetOrigin([0.,0.,0.])

    writer.SetInputData(mask)
    writer.SetFileName(mask_fn)
    writer.SetQFac(reader.GetQFac())
    q_mat = reader.GetQFormMatrix()
    writer.SetQFormMatrix(q_mat)
    s_mat = reader.GetSFormMatrix()
    writer.SetSFormMatrix(s_mat)
    writer.Write()
    return 

def write_vtk_image(vtkIm, fn, M=None):
    """
    This function writes a vtk image to disk
    Args:
        vtkIm: the vtk image to write
        fn: file name
    Returns:
        None
    """
    import vtk
    print("Writing vti with name: ", fn)
    if M is not None:
        M.Invert()
        reslice = vtk.vtkImageReslice()
        reslice.SetInputData(vtkIm)
        reslice.SetResliceAxes(M)
        reslice.SetInterpolationModeToNearestNeighbor()
        reslice.Update()
        vtkIm = reslice.GetOutput()

    _, extension = os.path.splitext(fn)
    if extension == '.vti':
        writer = vtk.vtkXMLImageDataWriter()
    elif extension == '.mhd':
        writer = vtk.vtkMetaImageWriter()
    else:
        raise ValueError("Incorrect extension " + extension)
    writer.SetInputData(vtkIm)
    writer.SetFileName(fn)
    writer.Update()
    writer.Write()
    return

def appendPolyData(poly_list):
    import vtk
    appendFilter = vtk.vtkAppendPolyData()
    for poly in poly_list:
        appendFilter.AddInputData(poly)
    appendFilter.Update()
    out = appendFilter.GetOutput() 
    return out

def cleanPolyData(poly, tol):
    import vtk
    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(poly)
    clean.SetTolerance(tol)
    clean.PointMergingOn()
    clean.Update()

    poly = clean.GetOutput()
    return poly


def load_vtk(fn, clean=True,num_mesh=1):
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
    poly = load_vtk_mesh(fn)    
    if clean:
        poly = cleanPolyData(poly, 0.)
    
    print("Loading vtk, number of cells and points: ", poly.GetNumberOfCells(), poly.GetNumberOfPoints())
    poly2_l = [poly]
    for i in range(1, num_mesh):
        poly2 = vtk.vtkPolyData()
        poly2.DeepCopy(poly)
        poly2_l.append(poly2)
    poly_f = appendPolyData(poly2_l)
    coords = vtk_to_numpy(poly_f.GetPoints().GetData())
    cells = vtk_to_numpy(poly_f.GetPolys().GetData())
    cells = cells.reshape(poly_f.GetNumberOfCells(), 4)
    cells = cells[:,1:]
    mesh = dict()
    mesh['faces'] = cells
    mesh['vertices'] = coords

    return mesh

def buildImageDataset(data_folder_out, modality, seed, mode='_train', ext='*.tfrecords'):
    import random
    x_train_filenames = []
    filenames = [None]*len(modality)
    nums = np.zeros(len(modality))
    for i, m in enumerate(modality):
        filenames[i] = glob.glob(os.path.join(data_folder_out, m+mode,ext))
        nums[i] = len(filenames[i])
        x_train_filenames+=filenames[i]
        #shuffle
        random.shuffle(x_train_filenames)
    random.shuffle(x_train_filenames)      
    print("Number of images obtained for training and validation: " + str(nums))
    return x_train_filenames

def construct_feed_dict(pkl):
    """Construct feed dictionary."""
    coord = pkl[0]
    pool_idx = pkl[4]
    #faces = pkl[5]
    # laplace = pkl[6]
    lape_idx = pkl[7]
    faces = pkl[8]
    edge_length = pkl[5]
    for i in range(1,4):
        adj = pkl[i][1]
    edges = pkl[6]
    feed_dict = dict()
    feed_dict['mesh_coords'] = coord
    feed_dict['edge_length'] = edge_length
    #feed_dict['mesh_coords'] = K.variable(coord, dtype=tf.float32)
    #feed_dict['pool_idxs'] = [tf.convert_to_tensor(pool_idx[i], dtype=tf.int32) for i in range(len(pool_idx))]
    feed_dict['lape_idx'] = [tf.convert_to_tensor(lape_idx[i], dtype=tf.int32) for i in range(len(lape_idx))]
    feed_dict['faces'] = [tf.convert_to_tensor(faces[i], dtype=tf.int32) for i in range(len(faces))]
    feed_dict['edges'] = [tf.convert_to_tensor(edges[i],dtype=tf.int32)  for i in range(len(edges))]
    feed_dict['adjs'] = [None]*3
    feed_dict['adjs'][0] = [tf.SparseTensor(indices=pkl[1][j][0], values=pkl[1][j][1].astype(np.float32), 
        dense_shape=(pkl[1][0][1].shape[0], pkl[1][0][1].shape[0])) for j in range(len(pkl[1]))]
    feed_dict['adjs'][1] = [tf.SparseTensor(indices=pkl[2][j][0], values=pkl[2][j][1].astype(np.float32), 
        dense_shape=(pkl[2][0][1].shape[0], pkl[2][0][1].shape[0])) for j in range(len(pkl[2]))]
    feed_dict['adjs'][2] = [tf.SparseTensor(indices=pkl[3][j][0], values=pkl[3][j][1].astype(np.float32), 
        dense_shape=(pkl[3][0][1].shape[0], pkl[3][0][1].shape[0])) for j in range(len(pkl[3]))]
       
    return feed_dict

def getTrainNLabelNames(data_folder, m, ext='*.nii.gz',fn='_train', seg_fn='_masks'):
  x_train_filenames = []
  y_train_filenames = []
  for subject_dir in sorted(glob.glob(os.path.join(data_folder,m+fn,ext))):
      x_train_filenames.append(os.path.realpath(subject_dir))
  try:
      for subject_dir in sorted(glob.glob(os.path.join(data_folder ,m+fn+seg_fn,ext))):
          y_train_filenames.append(os.path.realpath(subject_dir))
  except Exception as e: print(e)

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def data_to_tfrecords(X, Y, S,transform, spacing, file_path_prefix=None, verbose=True, debug=True):
           
    # Generate tfrecord writer
    result_tf_file = file_path_prefix + '.tfrecords'
           
    # Generate tfrecord writer
    result_tf_file = file_path_prefix + '.tfrecords'
    writer = tf.python_io.TFRecordWriter(result_tf_file)
    if verbose:
        print("Serializing example into {}".format(result_tf_file))
        
    # iterate over each sample,
    # and serialize it as ProtoBuf.
    
    d_feature = {}
    d_feature['X'] = _float_feature(X.flatten())
    d_feature['S'] = _int64_feature(S.flatten())

    if debug:
        print("**** X ****")
        print(X.shape, X.flatten().shape)
        print(X.dtype)
    for i, y in enumerate(Y):
        d_feature['Y_'+str(i)] = _float_feature(y.flatten())
        if debug:
            print("**** Y shape ****")
            print(y.shape, y.flatten().shape)
            print(y.dtype)

    d_feature['Transform'] = _float_feature(transform.flatten())
    d_feature['Spacing'] = _float_feature(spacing)
    #first axis is the channel dimension
    d_feature['shape0'] = _int64_feature([X.shape[0]])
    d_feature['shape1'] = _int64_feature([X.shape[1]])    
    d_feature['shape2'] = _int64_feature([X.shape[2]])

    center = np.mean(np.vstack(Y), axis=0)
    d_feature['center'] = _float_feature(center.flatten())
    if debug:
        print("***Center***")
        print(center)

    features = tf.train.Features(feature=d_feature)
    example = tf.train.Example(features=features)
    serialized = example.SerializeToString()
    writer.write(serialized)
    
    if verbose:
        print("Writing {} done!".format(result_tf_file))

def vtk_marching_cube(vtkLabel, bg_id, seg_id, smooth=None):
    import vtk
    contour = vtk.vtkDiscreteMarchingCubes()
    contour.SetInputData(vtkLabel)
    contour.SetValue(0, seg_id)
    contour.Update()
    mesh = contour.GetOutput()

    return mesh

def vtk_marching_cube_multi(vtkLabel, bg_id, smooth=None):
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    ids = np.unique(vtk_to_numpy(vtkLabel.GetPointData().GetScalars()))
    ids = np.delete(ids, np.where(ids==bg_id))

    contour = vtk.vtkDiscreteMarchingCubes()
    contour.SetInputData(vtkLabel)
    for index, i in enumerate(ids):
        print("Setting iso-contour value: ", i)
        contour.SetValue(index, i)
    contour.Update()
    mesh = contour.GetOutput()

    return mesh

def load_vtk_mesh(fileName):
    import vtk
    """
    Loads surface/volume mesh to VTK
    """
    if (fileName == ''):
        return 0
    fn_dir, fn_ext = os.path.splitext(fileName)
    if (fn_ext == '.vtk'):
        print('Reading vtk with name: ', fileName)
        reader = vtk.vtkPolyDataReader()
    elif (fn_ext == '.vtp'):
        print('Reading vtp with name: ', fileName)
        reader = vtk.vtkXMLPolyDataReader()
    elif (fn_ext == '.stl'):
        print('Reading stl with name: ', fileName)
        reader = vtk.vtkSTLReader()
    elif (fn_ext == '.vtu'):
        print('Reading vtu with name: ', fileName)
        reader = vtk.vtkXMLUnstructuredGridReader()
    elif (fn_ext == '.pvtu'):
        print('Reading pvtu with name: ', fileName)
        reader = vtk.vtkXMLPUnstructuredGridReader()
    else:
        print(fn_ext)
        raise ValueError('File extension not supported')

    reader.SetFileName(fileName)
    reader.Update()
    return reader.GetOutput()

def load_image_to_nifty(fn):
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    import SimpleITK as sitk
    _, ext = fn.split(os.extsep, 1)
    if ext=='vti':
        reader = vtk.vtkXMLImageDataReader()
        reader.SetFileName(fn)
        reader.Update()
        label_vtk = reader.GetOutput()
        size = label_vtk.GetDimensions()
        py_arr = np.reshape(vtk_to_numpy(label_vtk.GetPointData().GetScalars()), tuple(size), order='F')
        label = sitk.GetImageFromArray(py_arr.transpose(2,1,0))
        label.SetOrigin(label_vtk.GetOrigin())
        label.SetSpacing(label_vtk.GetSpacing())
        label.SetDirection(np.eye(3).ravel())
    elif ext=='nii' or ext=='nii.gz':
        label = sitk.ReadImage(fn)
    else:
        raise IOError("File extension is not recognized: ", ext)
    return label

def exportPython2VTK(img):
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, get_vtk_array_type

    vtkArray = numpy_to_vtk(num_array=img.flatten('F'), deep=True, array_type=get_vtk_array_type(img.dtype))
    return vtkArray



def write_vtk_polydata(poly, fn):
    import vtk
    print('Writing vtp with name:', fn)
    if (fn == ''):
        return 0

    _ , extension = os.path.splitext(fn)

    if extension == '.vtk':
        writer = vtk.vtkPolyDataWriter()
    elif extension == '.vtp':
        writer = vtk.vtkXMLPolyDataWriter()
    else:
        raise ValueError("Incorrect extension"+extension)
    writer.SetInputData(poly)
    writer.SetFileName(fn)
    writer.Update()
    writer.Write()
    return

def write_polydata_points(poly, fn):
    import vtk
    verts = vtk.vtkVertexGlyphFilter()
    verts.AddInputData(poly)
    verts.Update()
    write_vtk_polydata(verts.GetOutput(), fn)
    return

def write_numpy_points(pts, fn):
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk
    vtkPts = vtk.vtkPoints()
    vtkPts.SetData(numpy_to_vtk(pts[:,:3]))
    poly = vtk.vtkPolyData()
    poly.SetPoints(vtkPts)
    write_polydata_points(poly, fn)
    return 


def build_transform_matrix(image):
    matrix = np.eye(4)
    matrix[:-1,:-1] = np.matmul(np.reshape(image.GetDirection(), (3,3)), np.diag(image.GetSpacing()))
    matrix[:-1,-1] = np.array(image.GetOrigin())
    return matrix 

def get_point_normals(poly):
    import vtk
    norms = vtk.vtkPolyDataNormals()
    norms.SetInputData(poly)
    norms.ComputePointNormalsOn()
    norms.ComputeCellNormalsOff()
    norms.ConsistencyOn()
    norms.SplittingOff()
    norms.Update()
    poly = norms.GetOutput()
    pt_norm = poly.GetPointData().GetArray("Normals")
    from vtk.util.numpy_support import vtk_to_numpy
    return vtk_to_numpy(pt_norm)
