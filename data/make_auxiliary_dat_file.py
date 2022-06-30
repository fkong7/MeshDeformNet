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
import numpy as np
import networkx as nx
import scipy.sparse as sp
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../src'))
import pickle
import trimesh
import argparse
from scipy.sparse.linalg.eigen.arpack import eigsh
from scipy.sparse import csr_matrix
import vtk
from utils import load_vtk_mesh, write_vtk_polydata, cleanPolyData, load_vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import argparse

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))
    
    np.set_printoptions(threshold=sys.maxsize)
    
    return sparse_to_tuple(t_k)

def write_vtp(path, vertices, faces):
    poly = vtk.vtkPolyData()
    pts = vtk.vtkPoints()
    pts.SetData(numpy_to_vtk(vertices))
    poly.SetPoints(pts)

    tris = vtk.vtkCellArray()
    for i in range(faces.shape[0]):
        tri = vtk.vtkTriangle()
        tri.GetPointIds().SetId(0, faces[i][0])
        tri.GetPointIds().SetId(1, faces[i][1])
        tri.GetPointIds().SetId(2, faces[i][2])
        tris.InsertNextCell(tri)
    poly.SetPolys(tris)
    write_vtk_polydata(poly, path) 
    #poly = fix_polydata_normals(poly)
    #write_vtk_polydata(poly, path.split('.')[0]+'_flipped_n.vtp')

def cal_lap_index(mesh_neighbor):
    max_len = 0
    for j in mesh_neighbor:
        if len(j) > max_len:
            max_len = len(j)
    print("Max vertex degree is ", max_len)
    lap_index = np.zeros([mesh_neighbor.shape[0], max_len+2]).astype(np.int32)
    for i, j in enumerate(mesh_neighbor):
        lenj = len(j)
        lap_index[i][0:lenj] = j
        lap_index[i][lenj:-2] = -1
        lap_index[i][-2] = i
        lap_index[i][-1] = lenj
    return lap_index

def normalize(coords):
    """
    normalize the coordinates such that the maximum distance between surface and centroid
    is 1
    """
    coords -= np.mean(coords, axis=0)
    centroid = np.mean(coords, axis=0)
    dist = np.linalg.norm(coords - centroid, axis = 1)
    max_dist = dist[np.argmax(dist)]
    coords /= max_dist
    return coords

def fix_polydata_normals(poly):
    normAdj = vtk.vtkPolyDataNormals()
    normAdj.SetInputData(poly)
    normAdj.SplittingOff()
    normAdj.ConsistencyOn()
    normAdj.FlipNormalsOn()
    normAdj.Update()
    poly = normAdj.GetOutput()
    return poly

def create_data_aux(template_path, num_mesh, out_dir):
    TEMPLATE_NAME = 'sphere_coarse.vtp' 
    try:
        os.makedirs(out_dir)
    except Exception as e: print(e)
    info = {}
    info['coords'] = None
    info['support'] = {'stage1':None,'stage2':None,'stage3':None}
    info['unpool_idx'] = {'stage1_2':None,'stage2_3':None}
    info['lap_idx'] = {'stage1':None,'stage2':None,'stage3':None}
    info['faces'] = {'stage1': None, 'stage2':None,'stage3':None}
    info['edges'] = {'stage1': None, 'stage2':None,'stage3':None}
    
    # Simply load obj file created by Meshlab
    raw_mesh = load_vtk(template_path, num_mesh=num_mesh)
    raw_mesh_single = load_vtk(template_path, num_mesh=1)
    raw_mesh['vertices'] = normalize(raw_mesh['vertices'])
    raw_mesh_single['vertices'] = normalize(raw_mesh_single['vertices'])

    # Reload mesh using trimesh to get adjacent matrix, set `process=Flase` to preserve mesh vertices order
    print("Mesh info:  vertices: %d, faces: %d " % (raw_mesh['vertices'].shape[0], raw_mesh['faces'].shape[0]))
    mesh = trimesh.Trimesh(vertices=raw_mesh['vertices'], faces=(raw_mesh['faces']), process=False)
    assert np.all(raw_mesh['faces'] == mesh.faces)
    # ## Stage 1 auxiliary matrix
    coords_1 = np.array(mesh.vertices, dtype=np.float32)
    info['coords'] = coords_1
    adj_1 = nx.adjacency_matrix(mesh.vertex_adjacency_graph, nodelist=range(len(coords_1)))
    cheb_1 = chebyshev_polynomials(adj_1,1)
    info['support']['stage3'] = cheb_1
    
    mesh_single = trimesh.Trimesh(vertices=raw_mesh_single['vertices'], faces=(raw_mesh_single['faces']), process=False)
    assert np.all(raw_mesh_single['faces'] == mesh_single.faces)
    write_vtp(os.path.join(out_dir, 'init3.vtk'), mesh_single.vertices, mesh_single.faces) 
    info['faces']['stage3'] = raw_mesh_single['faces']

    lap_1 = cal_lap_index(np.array(mesh_single.vertex_neighbors))
    info['lap_idx']['stage3'] = lap_1
    coords_1_single = np.array(mesh_single.vertices, dtype=np.float32)
    adj_1_single = nx.adjacency_matrix(mesh_single.vertex_adjacency_graph, nodelist=range(len(coords_1_single)))
    cheb_1_single = chebyshev_polynomials(adj_1_single,1)
    edges = cheb_1_single[1][0] 
    # remove self loop
    edges = edges[edges[:,0] != edges[:,1],:]
    info['edges']['stage3'] = edges
    edge_length = np.sum(np.square(np.take(info['coords'],info['edges']['stage3'][:,0], axis=0) - np.take(info['coords'], info['edges']['stage3'][:,1], axis=0)),-1)
    edge_length = np.mean(edge_length)

    sparse_place_holder = sparse_to_tuple(csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]]))
    # ## Dump .dat file
    dat = [info['coords'],
           [sparse_place_holder, sparse_place_holder],           
           [sparse_place_holder, sparse_place_holder],           
           info['support']['stage3'],
           [np.zeros((1,4)), np.zeros((1,4))],
           [0, 0, edge_length],
           [np.zeros((1,4)), np.zeros((1,4)), info['edges']['stage3']],
           [np.zeros((1,4)), np.zeros((1,4)), info['lap_idx']['stage3']],
           [np.zeros((1,4)), np.zeros((1,4)), info['faces']['stage3']],
          ]
    print("Total number of vertices: ", len(info['coords']))
    pickle.dump(dat, open(os.path.join(out_dir, "data_aux.dat"),"wb"), protocol=2)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--template_fn', default='./template/sphere.vtp')
    parser.add_argument('--num_meshes', default=7, type=int)
    parser.add_argument('--output', default='./template')
    args = parser.parse_args()

    create_data_aux(args.template_fn, args.num_meshes, args.output)
