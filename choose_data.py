#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 00:13:02 2021

@author: user
"""

import os
import numpy as np
import torch.utils.data as torch_data
# import kitti_utils
import cv2
from PIL import Image
import transforms
import glob
from KITTI_to_pts import depth_to_pts
import pickle
from copy import deepcopy
from numpy.random import default_rng
import dataset.pointcloud_processor as pointcloud_processor
import torch

oheight, owidth =  352,1216        #352,1216
data_folder = '/mnt/5698b0ac-54a8-49e2-b934-ab1d9680a189/data/russell/self-supervised-depth-completion/data/'
val = 'select'

def load_calib():
    """
    Temporarily hardcoding the calibration matrix using calib file from 2011_09_26
    """
    calib = open("./calib_cam_to_cam.txt", "r")
    lines = calib.readlines()
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

def get_paths_and_transform(split):

    if split == "train":
        transform = train_transform
        glob_d = os.path.join(
            data_folder,
            'data_depth_velodyne/train/*_sync/proj_depth/velodyne_raw/image_0[2,3]/*.png'
        )
        glob_gt = os.path.join(
            data_folder,
            'data_depth_annotated/train/*_sync/proj_depth/groundtruth/image_0[2,3]/*.png'
        )

    elif split == "val":
        if val == "full":
            transform = val_transform
            glob_d = os.path.join(
                data_folder,
                'data_depth_velodyne/val/*_sync/proj_depth/velodyne_raw/image_0[2,3]/*.png'
            )
            glob_gt = os.path.join(
                data_folder,
                'data_depth_annotated/val/*_sync/proj_depth/groundtruth/image_0[2,3]/*.png'
            )

        elif val == "select":
            transform = val_transform
            glob_d = os.path.join(
                data_folder,
                "depth_selection/val_selection_cropped/velodyne_raw/*.png")
            glob_gt = os.path.join(
                data_folder,
                "depth_selection/val_selection_cropped/groundtruth_depth/*.png"
            )

    else:
        raise ValueError("Unrecognized split " + str(split))

    if glob_gt is not None:
        # train or val-full or val-select
        paths_d = sorted(glob.glob(glob_d)) 
        paths_gt = sorted(glob.glob(glob_gt)) 
        # paths_mask = [get_mask_paths(p) for p in paths_gt]
        # if split =='train':
        #     choice = rng.choice(len(paths_d), 42000, replace=False)
        #     choice = choice.tolist()
        #     paths_d = [paths_d[idx] for idx in choice]
        #     paths_gt = [paths_gt[idx] for idx in choice]
        #     paths_mask = [paths_mask[idx] for idx in choice]
        # else:
        #     choice = rng.choice(len(paths_d), 972, replace=False)
        #     choice = choice.tolist()
        #     paths_d = [paths_d[idx] for idx in choice]
        #     paths_gt = [paths_gt[idx] for idx in choice]
        #     paths_mask = [paths_mask[idx] for idx in choice]

    if len(paths_d) == 0 and len(paths_gt) == 0:
        raise (RuntimeError("Found 0 images under {}".format(glob_gt)))
    if len(paths_gt) != len(paths_d):
        raise (RuntimeError("Produced different sizes for datasets"))

    paths = { "d": paths_d, "gt": paths_gt}
    return paths, transform

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

    depth = depth_png.astype(np.float) / 256.
    # depth[depth_png == 0] = -1.
    depth = np.expand_dims(depth, -1)
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


def val_transform(sparse, target):
    transform = transforms.Compose([
        transforms.BottomCrop((oheight, owidth))
    ])
        
    if sparse is not None:
        sparse = transform(sparse)
    if target is not None:
        target = transform(target)
    return sparse, target

def train_transform(sparse, target):
    # s = np.random.uniform(1.0, 1.5) # random scaling
    # angle = np.random.uniform(-5.0, 5.0) # random rotation degrees
    # do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip
    do_flip = False
    transform_geometric = transforms.Compose([
        # transforms.Rotate(angle),
        # transforms.Resize(s),
        transforms.BottomCrop((oheight, owidth)),
        transforms.HorizontalFlip(do_flip)
    ])
    sparse = transform_geometric(sparse)
    target = transform_geometric(target)
    # sparse = drop_depth_measurements(sparse, 0.9)

    return sparse, target

split = 'val'
paths, transform = get_paths_and_transform(split)
K = load_calib()
fp_gt = open("./val_gt.txt", "a")
fp_pts = open("./val_pts.txt", "a")

count = 0
for index in range(len(paths['d'])):
    # print(index)
    sparse = depth_read(paths['d'][index])
    target = depth_read(paths['gt'][index])
    sparse, target = transform(sparse, target)
    points, idx = depth_to_pts(sparse, K)
    target, _ = depth_to_pts(target, K)
    idx = target[:,2]>80
    target = np.delete(target, idx, 0)
    # target = np.concatenate([target, points], axis = 0)
    # target = np.unique(target, axis=0)
    if target.shape[0]>=33616 and points.shape[0]>=16384:
        fp_gt.write(paths['gt'][index]+'\n')
        fp_pts.write(paths['d'][index]+'\n')
        count+=1
    print('index:', index, '    count:', count)
fp_gt.close()
fp_pts.close()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    