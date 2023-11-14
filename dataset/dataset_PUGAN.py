#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 12:53:32 2021

@author: user
"""

import os
import numpy as np
import logging
import torch.utils.data as torch_data
import torch
import h5py
import glob

def load_h5_data(h5_filename='', split='train'):
    logging.info("========== Loading Data ==========")
    # num_point = opts.num_point
    num_4X_point = 1024
    num_out_point = 1024

    logging.info("loading data from: {}".format(h5_filename))
    # if split == 'train':
    #     with h5py.File(h5_filename, 'r') as f:
    #         input = f['poisson_%d' % num_4X_point][:51000]
    #         gt = f['poisson_%d' % num_out_point][:51000]
    # else:
    #     with h5py.File(h5_filename, 'r') as f:
    #         input = f['poisson_%d' % num_4X_point][51000:]
    #         gt = f['poisson_%d' % num_out_point][51000:]
    with h5py.File(h5_filename, 'r') as f:
        input = f['poisson_%d' % num_4X_point][:]
        gt = f['poisson_%d' % num_out_point][:]

    # name = f['name'][:]
    assert len(input) == len(gt)

    logging.info("Normalize the data")
    # data_radius = np.ones(shape=(len(input)))
    centroid = np.mean(input[:, :, 0:3], axis=1, keepdims=True)
    input[:, :, 0:3] = input[:, :, 0:3] - centroid
    furthest_distance = np.amax(np.sqrt(np.sum(input[:, :, 0:3] ** 2, axis=-1)), axis=1, keepdims=True)
    input[:, :, 0:3] = input[:, :, 0:3] / np.expand_dims(furthest_distance, axis=-1)

    gt[:, :, 0:3] = gt[:, :, 0:3] - centroid
    gt[:, :, 0:3] = gt[:, :, 0:3] / np.expand_dims(furthest_distance, axis=-1)
    

    logging.info("total %d samples" % (len(input)))

    logging.info("========== Finish Data Loading ========== \n")
    return input, gt

def read_off(filename):
    # num_select=1024
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
    # if len(All_points) < (num_select + 3):
    #     return None
    # # take and shuffle points
    # index = np.random.choice(len(All_points), num_select, replace=False)
    # selected_points = All_points[index,:]
    return All_points

input_path = "./dataset/data/PUGAN/input/*.xyz"
gt_path = "./dataset/data/PUGAN/gt/*.xyz"

class PUGAN_Dataset(torch_data.Dataset):
    def __init__(self,opt, split='train' ):
        self.opt = opt
        self.split = split
        # paths, transform = get_paths_and_transform(split, opt)
        self.paths = opt.data_pugan
        if split == 'val':
            self.input_= sorted(glob.glob(input_path))
            self.gt = sorted(glob.glob(gt_path))
        else:
            self.input_, self.gt = load_h5_data(self.paths, self.split)
        # self.transform = transform
        # self.K = load_calib()
        # self.npoints = 8192
        # self.npoints = 16384
        print(len(self.input_))
        
        
    def __getitem__(self, index):
        if self.split == 'train':
            input_patch = self.input_[index, :, :]
            gt_patch = self.gt[index, :, :]
            choice = np.random.choice(input_patch.shape[0], 256, replace=False)
            input_patch = input_patch[choice,:]
            input_patch = torch.from_numpy(input_patch).float()
            gt_patch = torch.from_numpy(gt_patch).float()
        
        else:
            filename_gt = self.gt[index]
            filename_input = self.input_[index]
            input_patch = np.loadtxt(filename_input).astype(np.float32)
            gt_patch = np.loadtxt(filename_gt).astype(np.float32)
            
            # "Normalize the val data"

            # centroid = np.mean(input_patch[:, 0:3], axis=0, keepdims=True)
            # input_patch[:, 0:3] = input_patch[:, 0:3] - centroid
            # furthest_distance = np.amax(np.sqrt(np.sum(input_patch[:, 0:3] ** 2, axis=-1)), axis=0, keepdims=True)

            # input_patch[:, 0:3] = input_patch[:, 0:3] / np.expand_dims(furthest_distance, axis=-1)
        
            # gt_patch[:, 0:3] = gt_patch[:, 0:3] - centroid
            # gt_patch[:, 0:3] = gt_patch[:, 0:3] / np.expand_dims(furthest_distance, axis=-1)
            # input_patch = torch.from_numpy(input_patch).float()
            # gt_patch = torch.from_numpy(gt_patch).float()
        
        return_dict = {"points":input_patch, "target":gt_patch, "path": self.input_[index]}
        items = {
            key: val
            for key, val in return_dict.items() if val is not None
        }
        return items
        
    def __len__(self):
        return len(self.input_)