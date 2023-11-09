#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 20:32:35 2021

@author: user
"""

import torch
from auxiliary.my_utils import yellow_print
from model.model import EncoderDecoder
import torch.optim as optim
import numpy as np
import torch.nn as nn
from copy import deepcopy
import auxiliary.argument_parser as argument_parser
import dataset.dataset_KITTI as dataset_KITTI
from easydict import EasyDict
# from KITTI_to_pts import 

opt = argument_parser.parser()
opt.device = torch.device(f"cuda:{opt.multi_gpu[0]}")

datasets = EasyDict()
datasets.dataset_test = dataset_KITTI.KittiDataset(opt, 'val')

# calib = open("./calib_cam_to_cam.txt", "r")
# lines = calib.readlines()
# Rot = lines[21]
# Tran = lines[22]

# Rot_str = Rot.split(":")[1].split(" ")[1:]
# Rot_mat = np.reshape(np.array([float(p) for p in Rot_str]),
#                   (3, 3)).astype(np.float32)
# R = np.linalg.inv(Rot_mat[:3, :3])  # camera matrix
# # R = Rot_mat[:3, :3]  # camera matrix

# Tran_str = Tran.split(":")[1].split(" ")[1:]
# Tran_mat = np.reshape(np.array([float(p) for p in Tran_str]),
#                   (3, 1)).astype(np.float32)
# T = Tran_mat[:3, :]  # camera matrix
# T = np.transpose(T,(1,0))
# T = np.tile(T, 16384)
# T = np.reshape(T,(16384,3))

ang_res_y = 0.42

# min_total = 0
# max_total = 0
# total_angle = 
# for i in range(200):  
points = datasets.dataset_test[150]['points']
# points[:,[0,1,2]] = points[:,[2,0,1]]   

# points_lidar = np.matmul(points, R)
# points_lidar_ = points_lidar - T

# points_lidar = points - T
# points_lidar_ = np.matmul(points_lidar, R)


# x = points_lidar_[:,0]
# y = points_lidar_[:,1]
# z = points_lidar_[:,2]

y = points[:,0]
z = points[:,1]
x = points[:,2]

vertical_angle = np.arctan2(z, np.sqrt(x * x + y * y)) * 180.0 / np.pi
max_angle = np.max(vertical_angle)
min_angle = np.min(vertical_angle)
relative_vertical_angle = vertical_angle - min_angle
rowId = np.int_(np.round_(relative_vertical_angle / ang_res_y))
rowId = np.expand_dims(rowId, 1)
final = np.concatenate([points,rowId],axis = 1)
final_16 = final[final[:,3]%2==0,:]
pts = final.reshape(-1,1).astype('float32')
pts.tofile('./111.bin')

pts = final_16.reshape(-1,1).astype('float32')
pts.tofile('./111_16.bin')

    # if i ==0 :
    #     total_angle = vertical_angle
    # else:
    #     total_angle = np.concatenate([total_angle, vertical_angle], axis = 0)
#     min_i = np.min(vertical_angle)
#     max_i = np.max(vertical_angle)
#     min_total = min_total + min_i
#     max_total = max_total + max_i
# final_min = min_total/200
# final_max = max_total/200

relative_vertical_angle = vertical_angle +  14.4              #14.75

rowId = np.int_(np.round_(relative_vertical_angle / ang_res_y))
uni_ID = np.unique(rowId)

rowId = np.expand_dims(rowId, 1)

final = np.concatenate([points,rowId],axis = 1)
final_32 = final[final[:,3]%2==0,:]
final_32[:,[0,1,2]] = final_32[:,[1,2,0]]
points_lidar_[:,[0,1,2]] = points_lidar_[:,[1,2,0]]

# pts = points_lidar_.reshape(-1,1).astype('float32')
# pts.tofile('./1111.bin')

# pts = final_32.reshape(-1,1).astype('float32')
# pts.tofile('./111_32.bin')

# uni_ID = np.unique(rowId)
