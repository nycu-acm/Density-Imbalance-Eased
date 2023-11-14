import torch
import auxiliary.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D as dist_chamfer_3D


from auxiliary.ChamferDistancePytorch.fscore import fscore
import os
import training.metro as metro
from joblib import Parallel, delayed
import numpy as np
# from pointnet2 import pointnet2_utils
# from hausdorff import hausdorff_distance
# from auction_match import auction_match

# radius=0.5
# nsample=32
# radius=2.0
# nsample=250
# groupGT = pointnet2_utils.GroupGT(radius, nsample)
# furthest_point_sample = pointnet2_utils.FurthestPointSampling.apply
# gather_operation = pointnet2_utils.GatherOperation.apply



class TrainerLoss(object):
    """
    This class implements all functions related to the loss of Atlasnet, mainly applies chamfer and metro.
    Author : Thibault Groueix 01.11.2019
    """

    def __init__(self):
        super(TrainerLoss, self).__init__()

    def build_losses(self):
        """
        Create loss functions.
        """
        self.distChamfer = dist_chamfer_3D.chamfer_3DDist()
        self.loss_model = self.chamfer_loss

    def fuse_primitives(self):
        """
        Merge generated surface elements in a single one and prepare data for Chamfer
        Input size : batch, prim, 3, npoints
        Output size : batch, prim*npoints, 3
        :return:
        """
        self.data.pointsReconstructed = self.data.pointsReconstructed_prims.contiguous()
        # print(self.data.pointsReconstructed.shape)
        # self.data.pointsReconstructed_prims = self.data.pointsReconstructed_high.contiguous()   
        
        # self.data.pointsReconstructed_high = self.data.pointsReconstructed_high.contiguous()   
        # self.data.pointsReconstructed_high = self.data.pointsReconstructed_high.view(self.batch_size, -1, 3)
        # self.data.pointstile_high = self.data.pts_high.repeat(1,1,self.opt.number_points).reshape(self.batch_size,-1,3) 
        # self.data.pointsReconstructed_high = self.data.pointsReconstructed_high* self.opt.radius + self.data.pointstile_high
        
        # if not self.flags.train:
        #     self.data.pointsReconstructed_low = self.data.pointsReconstructed_low.contiguous()   
        #     self.data.pointsReconstructed_low = self.data.pointsReconstructed_low.view(self.batch_size, -1, 3)
        #     self.data.pointstile_low = self.data.pts_low.repeat(1,1,self.opt.number_points).reshape(self.batch_size,-1,3) 
        #     self.data.pointsReconstructed_low = self.data.pointsReconstructed_low* self.opt.radius*2 + self.data.pointstile_low
            
        #     self.data.pointsReconstructed = torch.cat([self.data.pointsReconstructed_high, self.data.pointsReconstructed_low, self.data.network_input],dim=1)

        # else:
        #     self.data.pointsReconstructed = torch.cat([self.data.pointsReconstructed_high, self.data.network_input],dim=1)


    def chamfer_loss(self):
        """
        Training loss of Atlasnet. The Chamfer Distance. Compute the f-score in eval mode.
        :return:
        """
        inCham1 = self.data.target.contiguous()
        inCham2 = self.data.pointsReconstructed.contiguous()
        # print(inCham1.shape, inCham2.shape)  # B N 3

        dist1, dist2, idx1, idx2 = self.distChamfer(inCham1, inCham2)  # mean over points
        # print(torch.max(dist1, dim=1)[0].shape)
        # print(torch.mean(dist1), torch.mean(dist2))
        self.data.loss = torch.mean(dist1) + torch.mean(dist2)  # mean over points
        if not self.flags.train:
            self.data.hausdorff = (torch.max(dist1,dim=1)[0] + torch.max(dist2,dim=1)[0])
            self.data.loss_fscore, _, _ = fscore(dist1, dist2)
            # print(self.data.loss_fscore)
            self.data.loss_fscore = self.data.loss_fscore.mean()
            # print(self.data.loss_fscore)
        if self.flags.train:
            # self.writer.add_scalar("chamfer_loss_total",self.data.loss, self.total_iteration)
            self.writer.add_scalar("chamfer_loss",self.data.loss, self.total_iteration)
            # self.writer.add_scalar("chamfer_loss_const",self.data.loss_const, self.total_iteration)

        # groupGT = pointnet2_utils.GroupGT(self.opt.radius, self.opt.nsample)
        # self.data.target_new = groupGT(self.data.target, self.data.pts_high)   # B 3 16384 32
        # self.data.target_new = self.data.target_new.permute(0,2,3,1).reshape(-1,self.opt.nsample,3)    #B 16384 32 3

        # inCham1 = self.data.target_new.view(self.data.target_new.size(0), -1, 3).contiguous()

        # inCham2 = self.data.pointsReconstructed_prims.contiguous()

        # dist1, dist2, idx1, idx2 = self.distChamfer(inCham1, inCham2) # mean over points

        # self.data.loss_local = (torch.mean(dist1) + torch.mean(dist2)) # mean over points

        # self.data.loss = self.data.loss_local

        # if not self.flags.train:
        #     # print(self.data.pointsReconstructed.shape)
        #     inCham3 = self.data.target.view(self.data.target.size(0), -1, 3).contiguous()
        #     inCham4 = self.data.pointsReconstructed.view(self.data.pointsReconstructed.size(0), -1, 3).contiguous()
        #     dist3, dist4, idx3, idx4 = self.distChamfer(inCham3, inCham4) # mean over points

        #     idx1 = dist3[0,:]<=0.01
        #     idx2 = dist4[0,:]<=0.01
        #     dist3 = dist3[0:1,idx1]

        #     dist4 = dist4[0:1,idx2]

        #     self.data.loss_glob = 0.5 * torch.mean(dist3) + 0.5 * torch.mean(dist4)  # mean over points
        #     self.data.loss_fscore, _, _ = fscore(dist3, dist4)
        #     self.data.loss_fscore = self.data.loss_fscore.mean()

        if self.flags.train:
            # self.writer.add_scalar("chamfer_loss_total",self.data.loss, self.total_iteration)
            self.writer.add_scalar("chamfer_loss_local",self.data.loss, self.total_iteration)
            # self.writer.add_scalar("chamfer_loss_const",self.data.loss_const, self.total_iteration)
            

    def metro(self):
        """
        Compute the metro distance on a randomly selected test files.
        Uses joblib to leverage as much cpu as possible
        :return:
        """
        metro_path = './dataset/data/metro_files'
        metro_files_path = '/'.join([metro_path, 'files-metro.txt'])
        self.metro_args_input = []
        if not os.path.exists(metro_files_path):
            os.system("chmod +x dataset/download_metro_files.sh")
            os.system("./dataset/download_metro_files.sh")
        ext = '.png' if self.opt.SVR else '.npy'
        with open(metro_files_path, 'r') as file:
            files = file.read().split('\n')

        for file in files:
            if file[-3:] == "ply":
                cat = file.split('/')[0]
                name = file.split('/')[1][:-4]
                input_path = '/'.join([metro_path, cat, name + ext])
                input_path_points = '/'.join([metro_path, cat, name + '.npy'])
                gt_path = '/'.join([metro_path, cat, name + '.ply'])
                path = self.demo(input_path, input_path_points)
                self.metro_args_input.append((path, gt_path))

        print("start metro calculus. This is going to take some time (30 minutes)")
        self.metro_results = Parallel(n_jobs=-1, backend="multiprocessing")(
            delayed(metro.metro)(*i) for i in self.metro_args_input)
        self.metro_results = np.array(self.metro_results).mean()
        print(f"Metro distance : {self.metro_results}")

    # def get_emd_loss(self,pred, gt):
    #     idx1 = pred[0,:,0]<10
    #     pred = pred[:,idx1,:].contiguous()
    #     idx2 = gt[0,:,0]<10
    #     gt = gt[:,idx2,:].contiguous()
    #     # print(pred.shape,gt.shape)
    #     idx, _ = auction_match(pred, gt)
    #     print(idx)
    #     matched_out = pointnet2_utils.gather_operation(gt.transpose(1, 2).contiguous(), idx)
    #     matched_out = matched_out.transpose(1, 2).contiguous()
    #     print(matched_out)
    #     dist2 = (pred - matched_out) ** 2
    #     dist2 = dist2.view(dist2.shape[0], -1) # <-- ???
    #     dist2 = torch.mean(dist2, dim=1, keepdims=True) # B,
    #     # dist2 /= pcd_radius
    #     print(torch.mean(dist2))
    #     return torch.mean(dist2)