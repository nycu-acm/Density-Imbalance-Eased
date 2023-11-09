from termcolor import colored
import torch
import numpy as np
# from pointnet2 import pointnet2_utils
import open3d as o3d
import cv2
import os
from KITTI_to_pts import convert_3d_to_2d
from pointnet2_ops import pointnet2_utils
from util import index_points, knn_point
from tqdm import tqdm

class TrainerIteration(object):
    """
        This class implements all functions related to a single forward pass of Atlasnet.
        Author : Thibault Groueix 01.11.2019
    """

    def __init__(self):
        super(TrainerIteration, self).__init__()

    def make_network_input(self):
        """
        Arrange to data to be fed to the network.
        :return:
        """
        if self.flags.train:  
            # self.data.network_input = self.data.points.contiguous().to(self.opt.device)
            # self.data.low1 = self.data.low1.contiguous().to(self.opt.device)
            # self.data.low2 = self.data.low2.contiguous().to(self.opt.device)
            # self.data.low3 = self.data.low3.contiguous().to(self.opt.device)
            # self.data.target = self.data.target.contiguous().to(self.opt.device)
            # print(self.data.network_input.shape)
            # print(self.data.low1.shape)
            # print(self.data.low2.shape)
            # print(self.data.low3.shape)
            
            # print(self.data.original.shape)   # B N 3
            X = sorted(np.random.randint(low=64, high=255, size=3))
            choice_high = np.random.choice(self.data.original.shape[1], 256, replace=False)
            choice_low1 = np.random.choice(self.data.original.shape[1], int(X[2]), replace=False)
            choice_low2 = np.random.choice(self.data.original.shape[1], int(X[1]), replace=False)
            choice_low3 = np.random.choice(self.data.original.shape[1], int(X[0]), replace=False)
            
            self.data.network_input = self.data.original[:,choice_high,:].contiguous().to(self.opt.device)
            self.data.low1 = self.data.original[:,choice_low1,:].contiguous().to(self.opt.device)
            self.data.low2 = self.data.original[:,choice_low2,:].contiguous().to(self.opt.device)
            self.data.low3 = self.data.original[:,choice_low3,:].contiguous().to(self.opt.device)
            self.data.target = self.data.target.contiguous().to(self.opt.device)
            # print(self.data.network_input.shape)
            # print(self.data.low1.shape)
            # print(self.data.low2.shape)
            # print(self.data.low3.shape)
            
            
            
            
        else:
            # self.data.network_input = self.data.points.contiguous().to(self.opt.device)
            # self.data.target = self.data.target.contiguous().to(self.opt.device)
            
            # seed_num = int(self.data.network_input.shape[1] / 256 *3)
            # # print(seed_num)
            # fps_idx = pointnet2_utils.furthest_point_sample(self.data.network_input, seed_num).long() # [B, npoint]
            # seed_xyz = index_points(self.data.network_input, fps_idx) 
            
            # knn_idx = knn_point(256, self.data.network_input, seed_xyz)
            # self.data.grouped_xyz = index_points(self.data.network_input, knn_idx).squeeze(0)  # 1, center_points, num_samples, C
            # # print(self.data.grouped_xyz.shape)
            ################################### PUGAN ##########################################
            # self.data.network_input = self.data.points.contiguous().to(self.opt.device)
            # self.data.target = self.data.target.contiguous().to(self.opt.device)
            
            # seed_num = int(self.data.network_input.shape[1] / 256 *3)
            # # print(seed_num)
            # fps_idx = pointnet2_utils.furthest_point_sample(self.data.network_input, seed_num).long() # [B, npoint]
            # seed_xyz = index_points(self.data.network_input, fps_idx) 
            
            # knn_idx = knn_point(256, self.data.network_input, seed_xyz)
            # self.data.grouped_xyz = index_points(self.data.network_input, knn_idx).squeeze(0)  # 1, center_points, num_samples, C
            ################################### PU1K ##########################################
            # self.data.network_input = self.data.points.contiguous().to(self.opt.device)
            # # calculate patch num
            
            # seed_num = int(self.data.points.shape[1] / 256 *3)
            # fps_idx = pointnet2_utils.furthest_point_sample(self.data.network_input, seed_num).long() # [B, npoint]
            # seed_xyz = index_points(self.data.network_input, fps_idx) 
            
            # # print("number of patches: %d" % seed_num)
            # knn_idx = knn_point(256, self.data.network_input, seed_xyz)
            # self.data.grouped_xyz = index_points(self.data.network_input, knn_idx).squeeze(0)  # 1, center_points, num_samples, C
            # self.data.target = self.data.target.contiguous().to(self.opt.device)   # 1, N, 3
            ################################### KITTI ##########################################
            
            self.data.network_input = self.data.points
            # print(self.data.network_input.shape)

            self.data.network_input = torch.from_numpy(self.data.network_input).to(self.opt.device).float()
            
            self.data.points = np.squeeze(self.data.points, axis=0)
            self.pcd = o3d.geometry.PointCloud()
            self.pcd.points = o3d.utility.Vector3dVector(self.data.points)
            _, clean_idx = self.pcd.remove_radius_outlier(nb_points=3, radius=1.0)
            self.data.clean = self.data.network_input[:, clean_idx, :]  # B N 3
            # self.data.clean = self.data.network_input.copy().contiguous()  # B N 3
            
            # calculate patch num
            
            seed_num = int(self.data.points.shape[0] / 256 *9)
            fps_idx = pointnet2_utils.furthest_point_sample(self.data.clean.contiguous(), seed_num).long() # [B, npoint]
            seed_xyz = index_points(self.data.clean, fps_idx) 
                        
            print("number of patches: %d" % seed_num)
            # knn_idx = knn_point(256, self.data.clean, seed_xyz)  # [B, S, nsample]
            # self.data.grouped_xyz = index_points(self.data.clean, knn_idx).squeeze(0)  # 1, center_points, num_samples, C
            ball_idx = pointnet2_utils.ball_query(2.0, 256, self.data.network_input, seed_xyz).long() # [B, npoint, nsample]
            self.data.grouped_xyz = index_points(self.data.network_input, ball_idx).squeeze(0)
            
            self.data.target = torch.from_numpy(self.data.target).to(self.opt.device).float()


    def common_ops(self):
        """
        Commom operations between train and eval forward passes
        :return:
        """
        self.make_network_input()
        if self.flags.train: 
            self.batch_size = self.data.network_input.size(0)
            # print(self.data.network_input.shape)   # B pts 3
            self.data.out_list, self.data.feat_list = self.network([self.data.network_input,self.data.low1,self.data.low2,self.data.low3],
                                                               train=self.flags.train)
            self.data.final_list=[]
            for i in range(len(self.data.out_list)):
                # print(self.data.out_list[i].shape)
                out_idx = pointnet2_utils.furthest_point_sample(self.data.out_list[i].contiguous(), 1024).long()
                final = index_points(self.data.out_list[i].contiguous(), out_idx) 
                self.data.final_list.append(final)
        # if self.flags.train:    
        #     self.batch_size = self.data.network_input.size(0)
        #     self.data.pointsReconstructed_prims = self.network(self.data.network_input,
        #                                                         train=self.flags.train)
        #     # print(self.data.pointsReconstructed_prims.shape)  # B 3 N
        #     self.data.pointsReconstructed_prims = self.data.pointsReconstructed_prims.contiguous().transpose(1,2)
        #     # print(self.data.pointsReconstructed_prims.shape)  32 1536  3
        #     out_idx = pointnet2_utils.furthest_point_sample(self.data.pointsReconstructed_prims.contiguous(), 1024).long()
        #     self.data.pointsReconstructed_prims = index_points(self.data.pointsReconstructed_prims.contiguous(), out_idx) 
        #     # print(self.data.pointsReconstructed_prims.shape)  32 1024 3
            
        else:
            self.outpatch_list = []
            patches = list(range(self.data.grouped_xyz.shape[0]))
            # for i in range(self.data.grouped_xyz.shape[0]):
            for i in tqdm(patches, total=len(patches)):
                patch = self.data.grouped_xyz[i:i+1,:,:]
                # print(patch.shape)   #  1 256 3
                patch, centroid, furthest_distance = self.normalize_point_cloud(patch)
                output_patch = self.network(patch.transpose(1,2), train=self.flags.train).transpose(1,2)  # 1 3 1024 --> 1 1024 3
                out_idx = pointnet2_utils.furthest_point_sample(output_patch.contiguous(), 1024).long()
                output_patch = index_points(output_patch.contiguous(), out_idx)
                output_patch = output_patch*furthest_distance + centroid
                output_patch, inverse = torch.unique(output_patch, sorted=False, return_inverse=True, dim=1)
                self.outpatch_list.append(output_patch)
        #############################################################################
            self.data.pointsReconstruct = torch.cat(self.outpatch_list, dim=1)   # B N 3
            # print(self.data.pointsReconstruct.shape)
            # fps_idx = pointnet2_utils.furthest_point_sample(self.data.pointsReconstruct, 8192).long() # [B, npoint]
            fps_idx = pointnet2_utils.furthest_point_sample(self.data.pointsReconstruct, self.data.points.shape[0]*4).long() # [B, npoint]
            # # print(fps_idx)
            self.data.pointsReconstructed_prims = index_points(self.data.pointsReconstruct.contiguous(), fps_idx)  # 1 N 3
            # # print("fps... ")
            # # self.data.pointsReconstructed_prims = index_points(self.data.pointsReconstruct, fps_idx)
            # # print(self.data.pointsReconstructed_prims.shape) # 1 3 8192
            # self.data.pointsReconstructed_prims, centroid, furthest_distance = self.normalize_point_cloud(self.data.pointsReconstructed_prims)
            # self.data.target,_ ,_ = self.normalize_point_cloud(self.data.target)
            # # self.data.target = self.normalize_ground_truth(self.data.target, centroid, furthest_distance)
            # # print("pred: ",self.data.pointsReconstructed_prims.shape)  # 1 N 3
            # # print("target: ",self.data.target.shape)   # 1 N 3
        #############################################################################
            # total_idx = np.arange(self.data.pointsReconstruct.shape[1])
            # back_idx = np.random.choice(total_idx, self.data.points.shape[0]*7, replace = False)
            # self.data.pointsReconstructed_prims = self.data.pointsReconstruct[:,back_idx,:]
            self.fuse_primitives()
        
        # print(self.data.pts.shape)
        
        # print("fuse prims... ")
        self.loss_model()  # batch
        # print("loss... ")
        
        # self.visualize()

    def train_iteration(self):
        """
        Forward backward pass
        :return:
        """
        self.optimizer.zero_grad()
        self.common_ops()
        # self.log.update("loss_train_total", self.data.loss.item())
        if not self.opt.no_learning:
            self.data.loss.backward()
            self.optimizer.step()  # gradient update
        self.print_iteration_stats(self.data.loss, self.data.cham_loss, self.data.smooth_loss)

    def visualize(self):
        if self.iteration % 50 == 1:
            tmp_string = "train" if self.flags.train else "test"
            self.visualizer.show_pointcloud(self.data.points[0], title=f"GT {tmp_string}")
            self.visualizer.show_pointcloud(self.data.pointsReconstructed[0], title=f"Reconstruction {tmp_string}")
            if self.opt.SVR:
                self.visualizer.show_image(self.data.image[0], title=f"Input Image {tmp_string}")

    def test_iteration(self):
        """
        Forward evaluation pass
        :return:
        """
        self.common_ops()
        # self.num_val_points = self.data.pointsReconstructed.size(1)
        # print(self.data.pointsReconstructed.shape)
        ##########################################################################################################################################################
        self.outputpoints = np.array(self.data.pointsReconstructed.squeeze(0).cpu(), dtype = np.float)
        # print(self.outputpoints.shape)  # 8192 3
        
        # self.outputpoints = np.unique(self.outputpoints,axis=0)
        self.zeros = np.zeros([self.outputpoints.shape[0],1])
        self.outputpoints_int = np.concatenate([self.outputpoints,self.zeros],axis=1)
        # if self.opt.create_depth:
        #     depth = convert_3d_to_2d(self.outputpoints,self.K)
        #     depth = depth*256
        #     depth = np.expand_dims(depth,axis = -1).astype(np.uint16)
        #     b = self.data.path[0].split('/')
        #     data_folder = '/'.join(b[:7]+['data_depth_high']+b[-6:-1])
        #     file_name = b[-1:]
        #     if not os.path.exists(data_folder):
        #         os.makedirs(data_folder)
        #     # data_folder = '/mnt/5698b0ac-54a8-49e2-b934-ab1d9680a189/data/russell/self-supervised-depth-completion/data/depth_selection/val_selection_cropped/velodyne_high/'
        #     new = '/'.join([data_folder]+file_name)
        #     # print(new)
        #     cv2.imwrite(new,depth)
        #############################################################################
        pts = self.outputpoints_int
        pts = pts.reshape(-1,1).astype('float32')
        # np.savetxt('./KITTI_3D/'+self.data.path[0].split('/')[-1].split('.')[0]+'.xyz', pts,fmt='%.6f')
        # pts.tofile('./KITTI_3D_train/'+self.data.path[0].split('/')[-1].split('.')[0]+'.bin')
        
        # np.savetxt('./patch/'+str(self.iteration)+'.xyz', pts, fmt='%.6f')
        # print(pts.shape)
        # np.savetxt('./PUGAN_pred/'+self.data.path[0].split('/')[-1], pts, fmt='%.6f')
        # np.savetxt('./CARLA_pred/'+self.data.path[0].split('/')[-1].split('.')[0]+'.xyz', pts,fmt='%.6f')
        # np.savetxt('./CARLA_pred/'+self.data.path.split('/')[-1].split('.')[0]+'.xyz', pts,fmt='%.6f')
        # np.savetxt('./CARLA_pred_3/'+self.data.path[0].split('/')[-1].split('.')[0]+'.xyz', pts,fmt='%.6f')
        #############################################################################
        # self.outputpoints_int[:,[0,1,2,3]] = self.outputpoints_int[:,[0,2,1,3]]
        # pts = self.outputpoints_int.reshape(-1,1).astype('float32')
        # pts.tofile('./output/'+str(self.iteration)+'.bin')
        #############################################################################
        # self.input = np.array(self.data.network_input.transpose(1,2).squeeze(0).cpu(), dtype = np.float)
        # print(self.input.shape)
        # np.savetxt('./input/'+self.data.path[0].split('/')[-1], self.input, fmt='%.6f')
        # cln_pts = self.input.reshape(-1,1).astype('float32')
        # cln_pts.tofile('./input/'+str(self.iteration)+'.bin')
        #############################################################################
        # self.target = np.array(self.data.target.transpose(1,2).squeeze(0).cpu(), dtype = np.float)
        # print(self.target.shape)
        # np.savetxt('./gt/'+self.data.path[0].split('/')[-1], self.target, fmt='%.6f')
        # gt = self.target.reshape(-1,1).astype('float32')
        # gt.tofile('./gt_PUGAN/'+str(self.iteration)+'.bin')
        #############################################################################
        
        ##########################################################################################################################################################
   
        self.log.update("chamfer_distance", self.data.loss.item())
        self.log.update("Hausdorff_distance", self.data.hausdorff.item())
        self.log.update("fscore", self.data.loss_fscore.item())
        
        print(
            '\r' + colored(
                '[%d: %d/%d]' % (self.epoch, self.iteration, self.datasets.len_dataset_test / self.opt.batch_size_test),
                'red') +
            colored('loss_val:  %f' % self.data.loss.item(), 'yellow'),
            end='')
        self.writer.add_scalar("loss_val",self.data.loss.item(), self.iteration)
    
    def normalize_point_cloud(self, input):
        """
        input: pc [N, P, 3]
        output: pc, centroid, furthest_distances
        """
        # print(input.shape)   # 1 256 3
        if len(input.shape) == 2:
            axis = 0
        elif len(input.shape) == 3:
            axis = 1
        # centroid = np.mean(input, axis=axis, keepdims=True)
        centroid = torch.mean(input, dim=axis, keepdim=True)
        # print(centroid.shape)  #  1 1 3
        
        input = input - centroid
        # furthest_distance = np.amax(
        #     np.sqrt(np.sum(input ** 2, axis=-1, keepdims=True)), axis=axis, keepdims=True)
        furthest_distance = torch.max(
            torch.sqrt(torch.sum(input**2, dim=-1, keepdims=True)), dim=axis, keepdim=True)
        # print(furthest_distance[0].shape) # 1 1 1
        input = input / furthest_distance[0]
        return input, centroid, furthest_distance[0]
    
    def normalize_ground_truth(self, input, centroid, furthest_distance):
        """
        input: pc [N, P, 3]
        output: pc, centroid, furthest_distances
        """
        # print(input.shape)   # 1 256 3
        # if len(input.shape) == 2:
        #     axis = 0
        # elif len(input.shape) == 3:
        #     axis = 1
        # # centroid = np.mean(input, axis=axis, keepdims=True)
        # centroid = torch.mean(input, dim=axis, keepdim=True)
        # # print(centroid.shape)  #  1 1 3
        
        input = input - centroid
        # furthest_distance = np.amax(
        #     np.sqrt(np.sum(input ** 2, axis=-1, keepdims=True)), axis=axis, keepdims=True)
        # furthest_distance = torch.max(
        #     torch.sqrt(torch.sum(input**2, dim=-1, keepdims=True)), dim=axis, keepdim=True)
        # print(furthest_distance[0].shape) # 1 1 1
        input = input / furthest_distance
        return input