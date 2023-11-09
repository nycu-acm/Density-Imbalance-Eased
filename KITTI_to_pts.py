#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 14:46:11 2021

@author: user
"""

from PIL import Image
import numpy as np
import os 
import transforms
import glob

def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    depth_png = np.array(img_file, dtype=int)
    # depth_png = np.resize(352,1216)
    img_file.close()
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert np.max(depth_png) > 255, \
        "np.max(depth_png)={}, path={}".format(np.max(depth_png),filename)

    # depth = depth_png.astype(np.float)
    depth = depth_png.astype(np.float) / 256.
    # depth[depth_png == 0] = -1.
    # depth = np.expand_dims(depth, -1)
    return depth

def mask_read(filename):
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename).convert('L')
    # rgb_png = np.array(img_file, dtype=float) / 255.0 # scale pixels to the range [0,1]
    mask_png = np.array(img_file, dtype='uint8')  # in the range [0,255]
    idx = mask_png!=0
    mask_png[idx]=1
    # rgb_png = np.resize(352,1216)
    img_file.close()
    return mask_png

oheight, owidth =  352,1216
transform_geometric = transforms.Compose([
        transforms.BottomCrop((oheight, owidth))])

def load_calib():
    """
    Temporarily hardcoding the calibration matrix using calib file from 2011_09_26
    """
    calib = open("./calib_cam_to_cam.txt", "r")
    lines = calib.readlines()
    # P_rect_line = lines[25]
    P_rect_line = lines[25]

    Proj_str = P_rect_line.split(":")[1].split(" ")[1:]
    Proj = np.reshape(np.array([float(p) for p in Proj_str]),
                      (3, 4)).astype(np.float32)
    K = Proj[:3, :3]  # camera matrix
    # note: we will take the center crop of the images during augmentation
    # that changes the optical centers, but not focal lengths
    K[0, 2] = K[
        0,
        2] - 13  # from width = 1242 to 1216, with a 13-pixel cut on both sides
    K[1, 2] = K[
        1,
        2] - 11.5  # from width = 375 to 352, with a 11.5-pixel cut on both sides
    return K

def convert_2d_to_3d(uv_depth , K):
    
    n = uv_depth.shape[0]
    v0 = K[1][2]
    u0 = K[0][2]
    fy = K[1][1]
    fx = K[0][0]
    y = -(uv_depth[:,0] - u0) * uv_depth[:,2]/ fx   #y
    z = -(uv_depth[:,1] - v0) * uv_depth[:,2]/ fy  #z
    pts_3d = np.zeros((n,3))
    # pts_3d[:,0]=x
    # pts_3d[:,1]=y
    # pts_3d[:,2]=uv_depth[:,2]
    pts_3d[:,0]=uv_depth[:,2]
    pts_3d[:,1]=y      # y 
    # pts_3d[:,2]=x
    pts_3d[:,2]=z     # z 
    return pts_3d

def convert_3d_to_2d(pts_3d , K):
    
    # n = uv_depth.shape[0]
    v0 = K[1][2]
    u0 = K[0][2]
    fy = K[1][1]
    fx = K[0][0]
    # x = (uv_depth[:,0] - u0) * uv_depth[:,2]/ fx
    pts_3d[:,1] = ((-pts_3d[:,1])*fy/pts_3d[:,0]) + u0    # y -> u
    # y = -((uv_depth[:,1] - v0) * uv_depth[:,2]/ fy)
    pts_3d[:,2] = ((-pts_3d[:,2])*fy/pts_3d[:,0]) + v0
    depth = np.ones((352,1216))*255
    for i in range(pts_3d.shape[0]):
        value = pts_3d[i,0]
        if round(pts_3d[i,1])>1215 or round(pts_3d[i,2])>351 or round(pts_3d[i,1])<0 or round(pts_3d[i,2])<0:
            pass
        elif value<depth[round(pts_3d[i,2]), round(pts_3d[i,1])]:
            depth[round(pts_3d[i,2]), round(pts_3d[i,1])] = value
    idx = depth==255
    depth[idx]=0
    # pts_3d[:,0]=x
    # pts_3d[:,1]=y
    # pts_3d[:,2]=uv_depth[:,2]
    return depth

def depth_to_pts(depth,K):
   # depth = depth_read(depth_img)
   # depth = transform_geometric(depth)
   # print(depth.shape)
   depth = np.squeeze(depth,2)
   u_map = np.arange(1216)
   u_map = np.tile(u_map,(352,1))
   v_map = np.arange(352)
   v_map = (np.tile(v_map,(1216,1))).transpose(1,0) 
   # print('u:',u_map.shape)
   # print('v:',v_map.shape)
   uvd = np.stack([u_map,v_map,depth],axis = 2)
   uvd_pts = uvd.reshape(-1,3)
   idx = uvd_pts[:,2]==0
   uvd_pts = np.delete(uvd_pts,idx,0)
   xyz_pts = convert_2d_to_3d(uvd_pts , K)
   #xyz_pts = xyz_pts.reshape(-1,1).astype('float32')
   return xyz_pts, idx

def file_to_pts(depth_img,K):
    depth = depth_read(depth_img)
    shape = depth.shape
    # print(shape)
    # depth = transform_geometric(depth)
    # u_map = np.arange(1216)
    # u_map = np.tile(u_map,(352,1))
    # v_map = np.arange(352)
    # v_map = (np.tile(v_map,(1216,1))).transpose(1,0) 
    
    u_map = np.arange(shape[1])
    u_map = np.tile(u_map,(shape[0],1))
    v_map = np.arange(shape[0])
    v_map = (np.tile(v_map,(shape[1],1))).transpose(1,0) 

    uvd = np.stack([u_map,v_map,depth],axis = 2)
    uvd_pts = uvd.reshape(-1,3)
    idx = uvd_pts[:,2]==0
    uvd_pts = np.delete(uvd_pts,idx,0)
    xyz_pts = convert_2d_to_3d(uvd_pts , K)
    #xyz_pts = xyz_pts.reshape(-1,1).astype('float32')
    return xyz_pts, idx

def high_to_low_res(points,ang_res_y):
    # print(points.shape)
    y = points[:,0]
    z = points[:,1]
    x = points[:,2]
    vertical_angle = np.arctan2(z, np.sqrt(x * x + y * y)) * 180.0 / np.pi
    # max_angle = np.max(vertical_angle)
    min_angle = np.min(vertical_angle)
    relative_vertical_angle = vertical_angle - min_angle
    rowId = np.int_(np.round_(relative_vertical_angle / ang_res_y))
    rowId = np.expand_dims(rowId, 1)
    idx = rowId[:,]%2==0
    # print(idx.shape)
    points = points[idx[:,0],:]
    
    return points


#--------------------------------------------------------------------------------------------#

# depth_img = "/mnt/5698b0ac-54a8-49e2-b934-ab1d9680a189/data/russell/self-supervised-depth-completion/data/depth_selection/val_selection_cropped/velodyne_raw/2011_09_26_drive_0005_sync_velodyne_raw_0000000056_image_02.png"
# # depth_img = "/mnt/5698b0ac-54a8-49e2-b934-ab1d9680a189/data/russell/self-supervised-depth-completion/data/data_depth_velodyne/train/2011_09_26_drive_0104_sync/proj_depth/velodyne_raw/image_02/0000000023.png"
# # depth_img = "/mnt/5698b0ac-54a8-49e2-b934-ab1d9680a189/data/russell/self-supervised-depth-completion/data/data_depth_annotated/train/2011_09_26_drive_0104_sync/proj_depth/groundtruth/image_02/0000000023.png"
# mask = "/mnt/5698b0ac-54a8-49e2-b934-ab1d9680a189/data/russell/self-supervised-depth-completion/data/data_car_seg/train/2011_09_26_drive_0001_sync/image_02/data/0000000005.png"
# # depth_img = "/mnt/5698b0ac-54a8-49e2-b934-ab1d9680a189/data/russell/self-supervised-depth-completion/data/data_rgb/train/2011_09_26_drive_0104_sync/image_02/data/0000000023.png"
# depth = depth_read(depth_img)
# depth = transform_geometric(depth)
# u_map = np.arange(1216)
# u_map = np.tile(u_map,(352,1))
# v_map = np.arange(352)
# v_map = (np.tile(v_map,(1216,1))).transpose(1,0)

# uvd = np.stack([u_map,v_map,depth],axis = 2)
# uvd_pts = uvd.reshape(-1,3)
# idx = uvd_pts[:,2]==0
# uvd_pts = np.delete(uvd_pts,idx,0)
# xyz_pts = convert_2d_to_3d(uvd_pts , K)
# mask = mask_read(mask)
# mask = transform_geometric(mask)
# mask = mask.reshape(-1,1)
# mask = np.delete(mask,idx,0)
# # print(mask.shape)
# # print(xyz_pts.shape)
# pts = np.concatenate([xyz_pts, mask], 1)
# print(pts.shape)
# pts = pts.reshape(-1,1).astype('float32')

# # with open('/mnt/5698b0ac-54a8-49e2-b934-ab1d9680a189/data/russell/AtlasNet/test.bin', 'w+b') as file:
# #     file.write(xyz_pts)


K = load_calib()
# depth_folder = '/mnt/5698b0ac-54a8-49e2-b934-ab1d9680a189/data/russell/AtlasNet/depthmap/*.png'
# depth_file = sorted(glob.glob(depth_folder))
# # depth_img = '/mnt/5698b0ac-54a8-49e2-b934-ab1d9680a189/data/russell/AtlasNet/depthmap/000001.png'
# for i in range(len(depth_file)):
#     pts,_ = file_to_pts(depth_file[i],K)
#     idx0 = pts[:,0]>5
#     idx1 = pts[:,0]<70 
#     idx2 = pts[:,2]>-40
#     idx3 = pts[:,2]<40
#     idx4 = pts[:,1]<=8
#     idx = idx0&idx1&idx2&idx3&idx4
#     # idx = idx1&idx4
#     pts = pts[idx,:]
#     zeros = np.zeros([pts.shape[0],1])
#     pts = np.concatenate([pts,zeros],axis=1)
#     pts = pts.reshape(-1,1).astype('float32')
#     pts.tofile('/mnt/5698b0ac-54a8-49e2-b934-ab1d9680a189/data/russell/AtlasNet/depth_3d/'+str(i).zfill(6)+'.bin')
# pts,_ = file_to_pts(depth_img,K)
# np.savetxt('./2011_09_26_drive_0005_sync_velodyne_raw_0000000056_image_02.xyz', pts,fmt='%.6f')
# idx1 = pts[:,0]<70 
# # idx2 = pts[:,1]>-40
# # idx3 = pts[:,1]<40
# idx4 = pts[:,1]<=8
# # idx = idx1&idx2&idx3&idx4
# idx = idx1&idx4
# pts = pts[idx,:]

# pts = pts.reshape(-1,1).astype('float32')
# pts.tofile('/mnt/5698b0ac-54a8-49e2-b934-ab1d9680a189/data/russell/AtlasNet/000001.bin')

# depth_img = "/mnt/5698b0ac-54a8-49e2-b934-ab1d9680a189/data/russell/self-supervised-depth-completion/output/0000000070.png"

#pts, _ = file_to_pts(depth_img,K)
#zeros = np.zeros([pts.shape[0],1])
#pts = np.concatenate([pts,zeros],axis=1)
#pts = pts.reshape(-1,1).astype('float32')
#pts.tofile('./0000000070.bin')
