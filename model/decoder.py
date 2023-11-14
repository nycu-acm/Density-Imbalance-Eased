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
        grids = np.meshgrid(np.linspace(-0.2, 0.2, 3, dtype=np.float32),
                            np.linspace(-0.2, 0.2, 2, dtype=np.float32))                               # (2, 4, 44)
        self.grids = torch.Tensor(grids).view(2, -1)  # (2, 4, 4) -> (2, 16)
    
    def forward(self, feat, points, train=True):
    # def forward(self, feat, train=True):
        # print(feat.shape)
        # feat shape  B 1024 256
        # grid 2 4
        # points B 3 num
        b = feat.size()[0]
        grids = self.grids.to(feat.device)
        # feat_tile = feat.transpose(1,2).repeat(1,1,4).view(b,-1,1024).transpose(1,2)
        feat_tile = feat.transpose(1,2).repeat(1,1,6).view(b,-1,1024).transpose(1,2)
        grid_feat = self.conv_grid(grids.unsqueeze(0)).repeat(1,1,points.shape[2])
        x = grid_feat +feat_tile
        # print(x.shape)
        x = self.activation(self.bn1(x))
        x = self.activation(self.bn2(self.conv2(x)))
        for i in range(2):
            x = self.activation(self.bn_list[i](self.conv_list[i](x)))
        out = self.last_conv(x)
        # print(out.shape)  # B 3 1024
        # points_tile = points.transpose(1,2).repeat(1,1,4).view(b,-1,3).transpose(1,2)
        points_tile = points.transpose(1,2).repeat(1,1,6).view(b,-1,3).transpose(1,2)
        final = out + points_tile
        return final
        # return out
    
        # b = feat.size()[0]
        # pts = feat.size()[1]
        
        # global features
        # v = feat  # (B, pts, 1024)

        # fully connected layers to generate the coarse output
        # x = F.relu(self.bn1(self.linear1(x)))
        # x = F.relu(self.bn2(self.linear2(x)))
        # x = self.linear3(x)
        # y_coarse = x.view(-1, 3, self.num_coarse)  # (B, 3, 1024)

        # repeated_centers = y_coarse.unsqueeze(3).repeat(1, 1, 1, 16).view(b, 3, -1)  # (B, 3, 16x1024)
        # repeated_v = v.unsqueeze(2).repeat(1, 1, 16 * self.num_coarse)               # (B, 1024, 16x1024)
        # repeated_centers = center.repeat(1,1,16).view(b,-1,3).transpose(2,1)           # (B, 3, 16*pts)
        # repeated_v = v.repeat(1,1,16).view(b,-1,1024).transpose(2,1)                   # (B, 1024, 16*pts)
        # grids = self.grids.to(feat.device)  # (2, 16)
        # grids = grids.unsqueeze(0).repeat(b, 1, pts)                     # (B, 2, 16*pts)

        # x = torch.cat([repeated_v, grids, repeated_centers], dim=1)                  # (B, 2+3+1024, 16x1024)
        # x = F.relu(self.bn3(self.conv1(x)))
        # x = F.relu(self.bn4(self.conv2(x)))
        # x = self.conv3(x)                # (B, 3, 16xpts)
        # y_detail = x*self.radius + repeated_centers  # (B, 3, 16x1024)

        # return x, y_detail, center