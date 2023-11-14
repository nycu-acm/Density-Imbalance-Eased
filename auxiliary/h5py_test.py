#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 15:33:12 2021

@author: user
"""

import h5py
import numpy as np
filename = "/mnt/5698b0ac-54a8-49e2-b934-ab1d9680a189/data/russell/PC_Transformer_multi_feature/dataset/data/PUGAN/PUGAN_poisson_256_poisson_1024.h5"

with h5py.File(filename, "r") as f:
    # List all groups
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[0]

    # Get the data
    data = list(f[a_group_key])
    
f = h5py.File(filename, 'r')

# point = np.fromfile("/mnt/5698b0ac-54a8-49e2-b934-ab1d9680a189/data/russell/PU-GAN/data/test1/fandisk.xyz", dtype=np.float32).reshape(-1, 4)[:, :3]

filename = "/mnt/5698b0ac-54a8-49e2-b934-ab1d9680a189/data/russell/PC_Transformer_multi_feature/dataset/data/PUGAN/test/elephant.off"

num_select=1024

def read_off(filename):
    num_select=1024
    f = open(filename)
    f.readline()
    f.readline()
    All_points = []
    # selected_points = []
    while True:
        new_line = f.readline().rstrip('\n')
        x = new_line.split(' ')
        while '' in x:
            x.remove('')
        if x[0] != '3':
            A = np.expand_dims(np.array(x[0:3], dtype='float32'),axis=1)
            All_points.append(A)
        else:
            break
    All_points = np.transpose(np.concatenate(All_points, axis=1),[1,0])
    # if the numbers of points are less than 2000, extent the point set
    if len(All_points) < (num_select + 3):
        return None
    # take and shuffle points
    index = np.random.choice(len(All_points), num_select, replace=False)
    selected_points = All_points[index,:]
    # for i in range(len(index)):
    #     selected_points.append(All_points[index[i]])
    return selected_points, All_points

select, All = read_off(filename)

num_select=1024
f = open(filename)
f.readline()
f.readline()
All_points = []
selected_points = []
while True:
    new_line = f.readline().rstrip('\n')
    x = new_line.split(' ')
    # x.remove('')
    while '' in x:
        x.remove('')
    if x[0] != '3':
        A = np.expand_dims(np.array(x[0:3], dtype='float32'),axis=1)
        All_points.append(A)
    else:
        break
# if the numbers of points are less than 2000, extent the point set
# if len(All_points) < (num_select + 3):
#     return None
# take and shuffle points
index = np.random.choice(len(All_points), num_select, replace=False)
for i in range(len(index)):
    selected_points.append(All_points[index[i]])
# return selected_points, All_points
out = np.concatenate(All, axis=1)
out = np.transpose(out, [1,0])
np.savetxt('./elephant.xyz', out, fmt='%.6f')


# points = read_off("/mnt/5698b0ac-54a8-49e2-b934-ab1d9680a189/data/russell/PU-GAN/data/test1/fandisk.off")

# def load(filename, count=None):
#     points = np.loadtxt(filename).astype(np.float32)
#     if count is not None:
#         if count > points.shape[0]:
#             # fill the point clouds with the random point
#             tmp = np.zeros((count, points.shape[1]), dtype=points.dtype)
#             tmp[:points.shape[0], ...] = points
#             tmp[points.shape[0]:, ...] = points[np.random.choice(
#                 points.shape[0], count - points.shape[0]), :]
#             points = tmp
#         elif count < points.shape[0]:
#             # different to pointnet2, take random x point instead of the first
#             # idx = np.random.permutation(count)
#             # points = points[idx, :]
#             points = downsample_points(points, count)
#     return points

# def read_off(file):
#     if 'OFF' != file.readline().strip():
#         raise('Not a valid OFF header')
#     n_verts, n_faces, n_dontknow = tuple([int(s) for s in file.readline().strip().split(' ')])
#     verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
#     faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
#     return verts, faces