import torch
import torch.nn as nn
from pointnet2.pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG
import pointnet2.pytorch_utils as pt_utils
import torch.nn.functional as F


def get_model(input_channels=0):
    return Pointnet2MSG(input_channels=input_channels)

#======================================== Original setting best =====================================================
# NSAMPLE = [[16]]
# MLPS = [[[64, 128, 1024]]]

# NPOINTS = [6144]
# RADIUS = [[0.5]]
#=======================================================================================================

NSAMPLE = [[16,32]]
MLPS = [[[64, 128, 1024],[64,128,1024]]]

NPOINTS = [6144]
RADIUS = [[0.5,1.0]]
#=======================================================================================================
FP_MLPS = [[128, 128]]
CLS_FC = [128]
DP_RATIO = 0.5


class Pointnet2MSG(nn.Module):
    def __init__(self, input_channels=0):
        super().__init__()

        self.SA_modules = nn.ModuleList()
        channel_in = input_channels     # 0

        skip_channel_list = [input_channels]
        # for k in range(NPOINTS.__len__()):    # 4
        mlps = MLPS[0].copy()
        channel_out = 0
        for idx in range(mlps.__len__()):
            mlps[idx] = [channel_in] + mlps[idx]
            channel_out += mlps[idx][-1]

        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=NPOINTS[0],
                radii=RADIUS[0],
                nsamples=NSAMPLE[0],
                mlps=mlps,
                use_xyz=True,
                bn=True
            )
        )
        # self.conv1 = torch.nn.Conv2d(128, 512, kernel_size=(1,1), stride = (1,1), bias=False)
        # self.conv2 = torch.nn.Conv2d(128, 512, kernel_size=(1,1), stride = (1,1), bias=False)
        # self.bn1 = torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # self.bn2 = torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.lin1 = nn.Linear(2048, 1024)
        self.lin2 = nn.Linear(1024, 1024)
        self.bn4 = torch.nn.BatchNorm1d(1024)
        self.bn5 = torch.nn.BatchNorm1d(1024)
        
        

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor, clean: torch.cuda.FloatTensor):
        xyz, features = self._break_up_pc(pointcloud)
        # print(xyz.shape)   # B, 16384, 3

        l_xyz, l_features = [xyz], [features]
        li_xyz, li_features = self.SA_modules[0](l_xyz[0] ,clean , l_features[0])
        # print(li_xyz.shape)   # B, 16384, 3
        # l_xyz.append(li_xyz)
        # l_features.append(li_features)
        # print(l_features[0])
        # feature = li_features[0]
        feature = li_features
        # print(feature.shape)   # B 1024 pts
        # feature2 = li_features[1]
        # print(feature1.shape)
        # print(feature2.shape)
        
        # feature1 = self.bn1(self.conv1(feature1))
        # feature2 = self.bn2(self.conv2(feature2))
        # feature = torch.stack([feature1, feature2], dim=1).transpose(2,1).contiguous()  # B, 2, 1024, 16384
        B = feature.shape[0]
        pts = feature.shape[2]
        print(feature.shape)
        feature = feature.permute(0,2,1).view(-1, 2048)
        feature = F.relu(self.bn4(self.lin1(feature).unsqueeze(-1)))
        feature = F.relu(self.bn5(self.lin2(feature.squeeze(2)).unsqueeze(-1))).squeeze(-1)
        # feature = feature.transpose(1,2)
        
        # feature = feature.reshape(B, pts, 1024).permute(0,2,1)
        feature = feature.reshape(B, pts, 1024)
        

        # pred_cls = self.cls_layer(l_features[0]).transpose(1, 2).contiguous()  # (B, N, 1)
        # print(l_features[0].shape)
        
        # return pred_cls, l_features[0].transpose(1, 2)
        return feature, li_xyz
