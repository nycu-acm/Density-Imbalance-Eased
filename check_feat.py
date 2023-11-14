#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 16:49:26 2021

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
import dataset.dataset_pu1k_xyz as dataset_pu1k
import dataset.dataset_carla_with_seg as dataset_KITTI
from easydict import EasyDict
from KITTI_to_pts import convert_3d_to_2d, load_calib
import os 
# from pointnet2 import pointnet2_utils
# from open3d import radius_outlier_removal 
# import open3d as o3d
# from pointconv import knn_point
# import pclpy 
# from pclpy import pcl
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

opt = argument_parser.parser()
device = torch.device(f"cuda:{1}")
network = EncoderDecoder(opt)
checkpoint = opt.reload_model_path
ckpt = torch.load(opt.reload_model_path)
network = torch.nn.DataParallel(network).cuda()
network.load_state_dict(ckpt)

datasets = EasyDict()
datasets.dataset_train = dataset_pu1k.PU1K_Dataset(opt, 'train')
# datasets.dataset_test = dataset_KITTI.KittiDataset(opt, 'val')
datasets.dataset_test = dataset_KITTI.KittiDataset('../PointRCNN/data', 'EVAL')
# datasets.dataloader_test = torch.utils.data.DataLoader(datasets.dataset_test,
#                                                        batch_size=opt.batch_size_test,
#                                                        shuffle=True, num_workers=int(opt.workers),
#                                                        collate_fn=datasets.dataset_test.collate_batch)

# for i in range(len(datasets.dataset_test)):
#     points = datasets.dataset_test[i]['points']
#     path = datasets.dataset_test[i]['path']
#     np.savetxt('./KITTI_val_xyz/'+path.split('/')[-1].split('.')[0]+'.xyz', points,fmt='%.6f')
points = datasets.dataset_test[0]['points']
target = datasets.dataset_test[0]['target']
from pointnet2_ops import pointnet2_utils
from util import index_points, knn_point
# points = torch.unsqueeze(points ,0)
# target = torch.unsqueeze(target ,0)

points = np.expand_dims(points ,0)
target = np.expand_dims(target ,0)
points = torch.from_numpy(points).to(device).float()
target = torch.from_numpy(target).to(device).float()

total_idx = np.arange(points.shape[1])
fps_idx = np.random.choice(total_idx, 100, replace = False)
select = points[:,fps_idx,:]

ball_idx = pointnet2_utils.ball_query(2.5, 256, points.contiguous(), select).long() # [B, npoint, nsample]
grouped_xyz = index_points(points, ball_idx).squeeze(0)

# fps_idx = pointnet2_utils.furthest_point_sample(points.contiguous(), 10).long()

# target = datasets.dataset_test[0]['target']
# network_input = torch.from_numpy(points).to(opt.device).float()    # B pts 3
# new = knn_point(5,network_input, network_input).int()
# print(new.shape)
# new_feature = pointnet2_utils.grouping_operation(network_input.transpose(2,1).contiguous(), new.contiguous())
# print(new_feature.shape)
# print(points.shape)
# outliers_idx = pointnet2_utils.furthest_point_sample(network_input, 1024).long()
# outliers_mask = torch.zeros([1,16384])
# outliers_mask[0, outliers_idx[0,:]]=1
# outliers_mask = outliers_mask.bool()
# clean = network_input[:,outliers_mask[0,:]!=True,:]

points = points.transpose(1,2).to(opt.device).float()

network = network.eval()
feature, final = network(points)
feature = np.array(feature.cpu().detach(), dtype=np.float)

# pointsReconstructed_prims, pts = network(network_input, clean)
# output_pts = np.array(pointsReconstructed_prims.cpu().detach(), dtype=np.float)
# pts = np.array(pts.cpu().detach(), dtype=np.float)

# pts = points.reshape(-1,1).astype('float32')
# pts.tofile('./11111.bin')

# #----------------------------------------------------------------------------------------------------------#
import numpy as np
import glob
from KITTI_to_pts import convert_3d_to_2d, load_calib, depth_read
import cv2
filepath_knn = "/mnt/5698b0ac-54a8-49e2-b934-ab1d9680a189/data/russell/PC_Transformer_multi_feature/KITTI_knn_pred/*.xyz"
filepath_ball = "/mnt/5698b0ac-54a8-49e2-b934-ab1d9680a189/data/russell/PC_Transformer_multi_feature/try/*.xyz"
filename_knn = sorted(glob.glob(filepath_knn))
filename_ball = sorted(glob.glob(filepath_ball))

knn = np.loadtxt(filename_knn[6]).astype(np.float64)
knn = np.transpose(knn, [1,0])
ball = np.loadtxt(filename_ball[6]).astype(np.float64)
knn_idx = knn[:,0]<=15
ball_idx = ball[:,0]>15
final = np.concatenate([knn[knn_idx[:,],:], ball[ball_idx[:,],: ]], axis=0)
filename_out = "/mnt/5698b0ac-54a8-49e2-b934-ab1d9680a189/data/russell/PC_Transformer_multi_feature/depth_combine/"+filename_ball[6].split('/')[-1]
print(filename_out)
np.savetxt(filename_out, final,fmt='%.6f')

K = load_calib()
filepath_combine = "/mnt/5698b0ac-54a8-49e2-b934-ab1d9680a189/data/russell/PC_Transformer_multi_feature/depth_combine/*.xyz"
filename_combine = sorted(glob.glob(filepath_combine))
combine = np.loadtxt(filename_combine[6]).astype(np.float64)
depth = convert_3d_to_2d(combine, K)
filepath_original = "/mnt/5698b0ac-54a8-49e2-b934-ab1d9680a189/data/russell/self-supervised-depth-completion/data/depth_selection/val_selection_cropped/velodyne_raw/2011_09_30_drive_0016_sync_velodyne_raw_0000000128_image_02.png"
original = depth_read(filepath_original)
idx = original!=0
depth[idx]=original[idx]
depth = depth*256
depth = np.expand_dims(depth,axis = -1).astype(np.uint16)
filename_depthmap = "/mnt/5698b0ac-54a8-49e2-b934-ab1d9680a189/data/russell/PC_Transformer_multi_feature/depth_map/"+filename_combine[6].split('/')[-1].split('.')[0]+".png"
cv2.imwrite(filename_depthmap,depth)




for i in range(len(filename_knn)):  
    knn = np.loadtxt(filename_knn[i]).astype(np.float64)
    knn = np.transpose(knn, [1,0])
    knn[:,[0,1,2]] = knn[:, [0,2,1]]
    ball = np.loadtxt(filename_ball[i]).astype(np.float64)
    knn_idx = knn[:,0]<=15
    ball_idx = ball[:,0]>15
    final = np.concatenate([knn[knn_idx,:], ball[ball_idx, :]], axis=0)
    filename_out = "/mnt/5698b0ac-54a8-49e2-b934-ab1d9680a189/data/russell/PC_Transformer_multi_feature/try_final/"+filename_ball[i].split('/')[-1]
    print(filename_out)
    np.savetxt(filename_out, final,fmt='%.6f')

filename_knn = "/mnt/5698b0ac-54a8-49e2-b934-ab1d9680a189/data/russell/PC_Transformer_multi_feature/KITTI_knn_pred/2011_09_26_drive_0002_sync_velodyne_raw_0000000005_image_02.xyz"
filename_ball = "/mnt/5698b0ac-54a8-49e2-b934-ab1d9680a189/data/russell/PC_Transformer_multi_feature/KITTI_ball_pred/2011_09_26_drive_0002_sync_velodyne_raw_0000000005_image_02.xyz"
filename_out = "/mnt/5698b0ac-54a8-49e2-b934-ab1d9680a189/data/russell/PC_Transformer_multi_feature/KITTI_val_xyz/2011_09_26_drive_0002_sync_velodyne_raw_0000000005_image_02.xyz"
knn = np.loadtxt(filename_knn).astype(np.float64)
ball = np.loadtxt(filename_ball).astype(np.float64)
knn_idx = knn[0,:]<=15
ball_idx = ball[0,:]>15
final = np.concatenate([knn[:,knn_idx], ball[:, ball_idx]], axis=1)
np.savetxt(filename_out, final,fmt='%.6f')


final = np.loadtxt(filename_out).astype(np.float64)
final = np.transpose(final, [1,0])
idx_0_10 = final[0,:]<=10
num_0_10 = sum(idx_0_10)

idx_0_20 = final[0,:]<=20
num_10_20 = sum(idx_0_20) - num_0_10

idx_0_30 = final[0,:]<=30
num_20_30 = sum(idx_0_30) - sum(idx_0_20)

idx_0_40 = final[0,:]<=40
num_30_40 = sum(idx_0_40) - sum(idx_0_30)

idx_0_50 = final[0,:]<=50
num_40_50 = sum(idx_0_50) - sum(idx_0_40)

idx_0_60 = final[0,:]<=60
num_50_60 = sum(idx_0_60) - sum(idx_0_50)

idx_0_70 = final[0,:]<=70
num_60_70 = sum(idx_0_70) - sum(idx_0_60)

idx_0_80 = final[0,:]<=80
num_70_80 = sum(idx_0_80) - sum(idx_0_70)

import matplotlib.pyplot as plt
# from pandas import Series,DataFrame
# import seaborn as sns
# import palettable
# from sklearn import datasets 

filename = "/mnt/5698b0ac-54a8-49e2-b934-ab1d9680a189/data/russell/PC_Transformer_multi_feature/KITTI_val_xyz/2011_09_26_drive_0013_sync_velodyne_raw_0000000041_image_03.xyz"
points = np.loadtxt(filename).astype(np.float64)
points[:,[0,1,2]]=points[:,[0,2,1]]
blank = np.zeros([16,20])
for i in reversed(range(1,17)):
    for j in range(-10,10):
        idx_x1 = points[:,0]<=5*i
        idx_x2 = points[:,0]>5*(i-1)
        idx_y1 = points[:,1]>=5*j
        idx_y2 = points[:,1]<=5*(j+1)
        blank[i-1,j+10]=sum(idx_x1 & idx_x2 & idx_y1 & idx_y2)
        
filename_pred = "/mnt/5698b0ac-54a8-49e2-b934-ab1d9680a189/data/russell/PC_Transformer_multi_feature/KITTI_final/2011_09_26_drive_0013_sync_velodyne_raw_0000000041_image_03.xyz"
points_pred = np.loadtxt(filename_pred).astype(np.float64)
points_pred = np.transpose(points_pred, [1,0])
points_pred[:,[0,1,2]]=points_pred[:,[0,2,1]]
blank_pred = np.zeros([16,20])
for i in reversed(range(1,17)):
    for j in range(-10,10):
        idx_x1 = points_pred[:,0]<=5*i
        idx_x2 = points_pred[:,0]>5*(i-1)
        idx_y1 = points_pred[:,1]>=5*j
        idx_y2 = points_pred[:,1]<=5*(j+1)
        blank_pred[i-1,j+10]=sum(idx_x1 & idx_x2 & idx_y1 & idx_y2)
        
plt.imshow(blank_pred, cmap=plt.get_cmap("gray"))

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
import numpy as np
filename_knn = "/mnt/5698b0ac-54a8-49e2-b934-ab1d9680a189/data/russell/PC_Transformer_multi_feature/KITTI_final/2011_09_26_drive_0013_sync_velodyne_raw_0000000041_image_03.xyz"
filename_ball = "/mnt/5698b0ac-54a8-49e2-b934-ab1d9680a189/data/russell/PC_Transformer_multi_feature/try/2011_09_26_drive_0013_sync_velodyne_raw_0000000041_image_03.xyz"
filename_out = "/mnt/5698b0ac-54a8-49e2-b934-ab1d9680a189/data/russell/PC_Transformer_multi_feature/try_result/2011_09_26_drive_0013_sync_velodyne_raw_0000000041_image_03.xyz"
knn = np.loadtxt(filename_knn).astype(np.float64)
knn = np.transpose(knn, [1,0])
ball = np.loadtxt(filename_ball).astype(np.float64)
ball[:,[0,1,2]] = ball[:,[0,2,1]]
knn_idx = knn[:,0]<=30
ball_idx = ball[:,0]>30
final = np.concatenate([knn[knn_idx,:], ball[ball_idx,:]], axis=0)
np.savetxt(filename_out, final,fmt='%.6f')

# # pcd = o3d.io.read_point_cloud("/mnt/5698b0ac-54a8-49e2-b934-ab1d9680a189/data/russell/AtlasNet/000001_gt.bin")
# # o3d.visualization.draw_geometries([pcd])

# points_save = np.squeeze(points, axis=0)
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points_save)
# # o3d.io.write_point_cloud("./data.ply", pcd)

# _, high_dense = pcd.remove_radius_outlier(nb_points=16, radius=0.5)

# high_pts = points_save[high_dense,:]

# _, low_dense = pcd.remove_radius_outlier(nb_points=16, radius=2.0)

# low_dense = sorted(list(set(low_dense) - set(high_dense)))
# # points_rmv = np.array(pcd.points)
# # o3d.visualization.draw_geometries([pcd])
# low_pts = points_save[low_dense,:]

# # pcd1 = o3d.geometry.PointCloud()
# # pcd1.points = o3d.utility.Vector3dVector(points_out)

# # _, b = pcd1.remove_radius_outlier(nb_points=16, radius=2.0)

# # points_final = points_out[b,:]

# # pts = points_out.reshape(-1,1).astype('float32')
# pts.tofile('./11111_16.bin')


# # total_pts = [i for i in range(16384)]
# # loose_pts = sorted(list(set(total_pts) - set(a)))

# ##------------------------------------------------------------------------------------------------------------------
# import torch
# from auxiliary.my_utils import yellow_print
# from model.model import EncoderDecoder
# import torch.optim as optim
# import numpy as np
# import torch.nn as nn
# from copy import deepcopy
# import auxiliary.argument_parser as argument_parser
# import dataset.dataset_KITTI as dataset_KITTI
# from easydict import EasyDict
# from pointnet2 import pointnet2_utils
# # from open3d import radius_outlier_removal 
# import open3d as o3d

# opt = argument_parser.parser()
# opt.device = torch.device(f"cuda:{opt.multi_gpu[0]}")
# network = EncoderDecoder(opt)
# network = nn.DataParallel(network, device_ids=opt.multi_gpu)
# # checkpoint = "./log/02021-02-19T22:13:08.090880/0_network.pth"
# ckpt = torch.load(opt.reload_model_path, map_location='cuda:0')

# # lidar_file = "/mnt/5698b0ac-54a8-49e2-b934-ab1d9680a189/data/russell/Indoor_lidar/0312/2021-03-12-15-45-28_Velodyne-VLP-16-Data (Frame 0638).bin"
# lidar_file = "/mnt/5698b0ac-54a8-49e2-b934-ab1d9680a189/data/russell/outdoor_lidar/1617082743938570000.bin"
# # lidar_file = "/mnt/5698b0ac-54a8-49e2-b934-ab1d9680a189/data/russell/PointRCNN/data/KITTI/object/training/velodyne_new_int/000005.bin"
# points_indoor = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
# points = points_indoor[:,:-1]
# # points[:,[0,1,2]]= points[:,[1,2,0]]
# # points = np.expand_dims(points,axis = 0)
# # points = torch.from_numpy(points).float()
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points)
# _, clean_idx = pcd.remove_radius_outlier(nb_points=3, radius=1.0)

# clean_points = points[clean_idx,:]
# pcd.points = o3d.utility.Vector3dVector(clean_points)


# radii_list = [1.9,1.6,1.3,1.0,0.7]
# total_list = [i for i in range(clean_points.shape[0])]
# idx_list = [total_list]
# for i in range(len(radii_list)):
# 	rad = radii_list[i]
# 	print(rad)
# 	_, idx = pcd.remove_radius_outlier(nb_points=16, radius=rad)
# # 	idx_curr = sorted(list(set(idx_list[i]) - set(idx)))
# 	idx_list.append(idx)
# 	
# idx_layer = []
# for i in range(len(idx_list)-1):
# 	idx = sorted(list(set(idx_list[i]) - set(idx_list[i+1])))
# 	idx_layer.append(idx)
# idx_layer.append(idx_list[-1])

# high_pts = points[high_dense,:]
# high_pts[:,[0,1,2]] = high_pts[:,[0,2,1]]
# pts = high_pts.reshape(-1,1).astype('float32')
# pts.tofile('./radii_0.7_16.bin')

# high_pts = np.expand_dims(high_pts,axis = 0)
# high_pts = torch.from_numpy(high_pts).to(opt.device).float()
# points = np.expand_dims(points,axis = 0)
# points = torch.from_numpy(points).to(opt.device).float()


# # network = torch.nn.DataParallel(network).cuda()
# network.load_state_dict(ckpt)

# pointsReconstructed_prims, pts = network(points, high_pts, train=False)

# pointsReconstructed = pointsReconstructed_prims.contiguous()   
# pointsReconstructed = pointsReconstructed.view(1, -1, 3)
# pointsReconstructed = pointsReconstructed

# pointstile = pts.repeat(1,1,9).reshape(1,-1,3) 

# pointsReconstructed = pointsReconstructed + pointstile
# pointsReconstructed = torch.cat([pointsReconstructed, points],dim=1)

# outputpoints = np.array(pointsReconstructed.squeeze(0).cpu().detach(), dtype = np.float)

# outputpoints = np.unique(outputpoints,axis=0)
# pts = outputpoints.reshape(-1,1).astype('float32')
# pts.tofile('./test.bin')

# ##----------------------------------------------------------------------------------------------------##

# lidar_file = "/mnt/5698b0ac-54a8-49e2-b934-ab1d9680a189/data/russell/PointRCNN/data/KITTI/object/training/velodyne_original_reduced/000009.bin"
# points_original = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
# points = points_original[:,:-1]
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points)
# _, clean_idx = pcd.remove_radius_outlier(nb_points=2, radius=0.3)
# clean = points[clean_idx,:]
# pcf = o3d.geometry.PointCloud()
# pcf.points = o3d.utility.Vector3dVector(clean)
# _, high_idx = pcf.remove_radius_outlier(nb_points=16, radius=0.5)
# clean_out = clean.reshape(-1,1).astype('float32')
# clean_out.tofile('./clean.bin')










