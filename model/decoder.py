#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 17:17:31 2021

@author: user
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(argument):
    getter = {
        "relu": F.relu,
        "sigmoid": F.sigmoid,
        "softplus": F.softplus,
        "logsigmoid": F.logsigmoid,
        "softsign": F.softsign,
        "tanh": F.tanh,
    }
    return getter.get(argument, "Invalid activation")

class Decoder(nn.Module):
    def __init__(self, opt):
        super(Decoder, self).__init__()
        
        self.opt = opt

        self.conv_grid = nn.Conv1d(2, 1024, 1)
        self.conv2 = nn.Conv1d(1024, 512, 1)
        
        self.bn1 = torch.nn.BatchNorm1d(1024)
        self.bn2 = torch.nn.BatchNorm1d(512)
        
        self.conv_list = nn.ModuleList(
            [torch.nn.Conv1d(512, 512, 1) for i in range(2)])
        
        self.bn_list = nn.ModuleList([torch.nn.BatchNorm1d(512) for i in range(2)])
        
        self.last_conv = torch.nn.Conv1d(512, 3, 1)

        self.activation = get_activation(opt.activation)
        
        # 2D grid
        # grids = np.meshgrid(np.linspace(-0.2, 0.2, 2, dtype=np.float32),
        #                     np.linspace(-0.2, 0.2, 2, dtype=np.float32))                               # (2, 4, 44)
        # # grids = np.meshgrid(np.linspace(-0.2, 0.2, 3, dtype=np.float32),
        # #                     np.linspace(-0.2, 0.2, 2, dtype=np.float32))                               # (2, 4, 44)
        # self.grids = torch.Tensor(grids).view(2, -1)  # (2, 4, 4) -> (2, 16)
        self.grid_high=torch.tensor(self.gen_grid(6)).cuda()
        self.grid_mid=torch.tensor(self.gen_grid(12)).cuda()
        self.grid_low=torch.tensor(self.gen_grid(20)).cuda()
        
    
    def forward(self, feature_list, train=True):
        # print(feat.shape)
        # feat shape  B 1024 256
        # grid 2 4
        # points B 3 num
        
        if train==True:
            coord_list=[]
            for i in range(len(feature_list)):
                feat = feature_list[i]
                b = feat.size()[0]
                if feat.shape[2]>=192:
                    grids = self.grid_high.to(feat.device)
                    # print(grids.shape)
                    feat_tile = feat.transpose(1,2).repeat(1,1,6).view(b,-1,1024).transpose(1,2)
                    # feat_tile = feat.transpose(1,2).repeat(1,1,6).view(b,-1,1024).transpose(1,2)
                    grid_feat = self.conv_grid(grids.unsqueeze(0).transpose(1,2)).repeat(1,1,feat.shape[2])
                elif feat.shape[2]<192 and feat.shape[2]>=96:
                    grids = self.grid_mid.to(feat.device)
                    feat_tile = feat.transpose(1,2).repeat(1,1,12).view(b,-1,1024).transpose(1,2)
                    # feat_tile = feat.transpose(1,2).repeat(1,1,6).view(b,-1,1024).transpose(1,2)
                    grid_feat = self.conv_grid(grids.unsqueeze(0).transpose(1,2)).repeat(1,1,feat.shape[2])
                else:
                    grids = self.grid_low.to(feat.device)
                    feat_tile = feat.transpose(1,2).repeat(1,1,20).view(b,-1,1024).transpose(1,2)
                    # feat_tile = feat.transpose(1,2).repeat(1,1,6).view(b,-1,1024).transpose(1,2)
                    grid_feat = self.conv_grid(grids.unsqueeze(0).transpose(1,2)).repeat(1,1,feat.shape[2])
                x = grid_feat +feat_tile
                # print(x.shape)
                x = self.activation(self.bn1(x))
                x = self.activation(self.bn2(self.conv2(x)))
                for i in range(2):
                    x = self.activation(self.bn_list[i](self.conv_list[i](x)))
                out = self.last_conv(x)
                # print(out.shape)  # 1 3 B
                coord_list.append(out.transpose(2,1))
            return coord_list
        else:
            feat = feature_list
            b = feat.size()[0]
            # grids = self.grid_high.to(feat.device)
            # # print(feat.shape)
            # feat_tile = feat.transpose(1,2).repeat(1,1,6).view(b,-1,1024).transpose(1,2)
            # grid_feat = self.conv_grid(grids.unsqueeze(0).transpose(1,2)).repeat(1,1,feat.shape[2])
            if feat.shape[2]>=192:
            	grids = self.grid_high.to(feat.device)
            	feat_tile = feat.transpose(1,2).repeat(1,1,6).view(b,-1,1024).transpose(1,2)
                    # feat_tile = feat.transpose(1,2).repeat(1,1,6).view(b,-1,1024).transpose(1,2)
            	grid_feat = self.conv_grid(grids.unsqueeze(0).transpose(1,2)).repeat(1,1,feat.shape[2])
            elif feat.shape[2]<192 and feat.shape[2]>=96:
                grids = self.grid_mid.to(feat.device)
                feat_tile = feat.transpose(1,2).repeat(1,1,12).view(b,-1,1024).transpose(1,2)
                    # feat_tile = feat.transpose(1,2).repeat(1,1,6).view(b,-1,1024).transpose(1,2)
                grid_feat = self.conv_grid(grids.unsqueeze(0).transpose(1,2)).repeat(1,1,feat.shape[2])
            else:
                grids = self.grid_high.to(feat.device)
                # print("else")
                feat_tile = feat.transpose(1,2).repeat(1,1,20).view(b,-1,1024).transpose(1,2)
                    # feat_tile = feat.transpose(1,2).repeat(1,1,6).view(b,-1,1024).transpose(1,2)
                grid_feat = self.conv_grid(grids.unsqueeze(0).transpose(1,2)).repeat(1,1,feat.shape[2])
            x = grid_feat +feat_tile
            # print(x.shape)
            x = self.activation(self.bn1(x))
            x = self.activation(self.bn2(self.conv2(x)))
            for i in range(2):
                x = self.activation(self.bn_list[i](self.conv_list[i](x)))
            out = self.last_conv(x)
            # print(out.shape)  # B 3 1024
            return out

    def gen_grid(self,up_ratio):
        import math
        sqrted=int(math.sqrt(up_ratio))+1
        for i in range(1,sqrted+1).__reversed__():
            if (up_ratio%i)==0:
                num_x=i
                num_y=up_ratio//i
                break
        print("num_x: ", num_x)
        print("num_y: ", num_y)
        grid_x=torch.linspace(-0.2,0.2,num_x)
        grid_y=torch.linspace(-0.2,0.2,num_y)
        
    
        x,y=torch.meshgrid([grid_x,grid_y])
        grid=torch.stack([x,y],dim=-1)#2,2,2
        grid=grid.view([-1,2])#4,2
        return grid