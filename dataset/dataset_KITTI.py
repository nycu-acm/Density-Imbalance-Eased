import os
import numpy as np
import torch.utils.data as torch_data
# import kitti_utils
import cv2
from PIL import Image
import transforms
import glob
from KITTI_to_pts import depth_to_pts, high_to_low_res
import pickle
from copy import deepcopy
from numpy.random import default_rng
import dataset.pointcloud_processor as pointcloud_processor
import torch
# from pointnet2 import pointnet2_utils

oheight, owidth =  352,1216        #352,1216
rng = default_rng()
normalization_function = pointcloud_processor.Normalization.normalize_unitL2ball_functional
normalization_self = pointcloud_processor.Normalization.normalize_unitL2ball_self
# pointnet2_utils.furthest_point_sample(xyz, self.npoint)

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

def get_paths_and_transform(split, opt):

    if split == "train":
        transform = train_transform
        text_d = open('./train_pts.txt')
        paths_d = text_d.readlines()
        text_d.close()
        text_gt = open('./train_gt.txt')
        paths_gt = text_gt.readlines()
        text_gt.close()
    elif split == "val":
        if opt.val == "full":
            transform = val_transform
            glob_d = os.path.join(
                opt.data_folder,
                'data_depth_velodyne/val/*_sync/proj_depth/velodyne_raw/image_0[2,3]/*.png'
            )
            glob_gt = os.path.join(
                opt.data_folder,
                'data_depth_annotated/val/*_sync/proj_depth/groundtruth/image_0[2,3]/*.png'
            )
        elif opt.val == "select":
            transform = val_transform
            text_d = open('./val_pts_small.txt')
            paths_d = text_d.readlines()
            text_d.close()
            text_gt = open('./val_gt_small.txt')
            paths_gt = text_gt.readlines()
            text_gt.close()
            #########################################################################################################
            # transform = val_transform
            # glob_d = os.path.join(
            #     opt.data_folder,
            #     "depth_selection/val_selection_cropped/velodyne_raw/*.png")
            # glob_gt = os.path.join(
            #     opt.data_folder,
            #     "depth_selection/val_selection_cropped/groundtruth_depth/*.png"
            # )
            #########################################################################################################
            # paths_d = sorted(glob.glob(glob_d)) 
            # paths_gt = sorted(glob.glob(glob_gt)) 

    else:
        raise ValueError("Unrecognized split " + str(split))


    if len(paths_d) == 0 and len(paths_gt) == 0:
        raise (RuntimeError("Found 0 images under {}".format(glob_gt)))
    if len(paths_gt) != len(paths_d) :
        raise (RuntimeError("Produced different sizes for datasets"))

    paths = {"d": paths_d, "gt": paths_gt}
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


def val_transform(sparse, target, opt):
    transform = transforms.Compose([
        transforms.BottomCrop((oheight, owidth))
    ])

    if sparse is not None:
        sparse = transform(sparse)
    if target is not None:
        target = transform(target)
    return sparse, target

def train_transform(sparse, target, opt):
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

to_tensor = transforms.ToTensor()
to_float_tensor = lambda x: to_tensor(x).float()
# point_list = [0,20000,40000,60000,80000,85898]

class KittiDataset(torch_data.Dataset):
    def __init__(self,opt, split='train' ):
        self.opt = opt
        self.split = split
        paths, transform = get_paths_and_transform(split, opt)
        self.paths = paths
        self.transform = transform
        self.K = load_calib()
        # self.npoints = 8192
        self.npoints = 16384
        print(len(self.paths['d']))
        # self.preprocess()
    
    # def preprocess(self):
    #     if os.path.exists('./data/'+self.split+'_points.pkl'):
    #         print('-------------------- load point cloud data --------------------')
    #         with open('./data/'+self.split+'_points.pkl', "rb") as fp:
    #             self.data_metadata = pickle.load(fp)
    #     else :
            
    #         print("preprocess dataset...")
    #         self.datas = [self.__getraw__(i) for i in range(self.__len__())]
            
    #         self.data_metadata = [{'points': a[0], 'target': a[1]} for a in self.datas]
            
    #         # Save in cache
    #         with open('./data/'+self.split+'_points.pkl', "wb") as fp:  # Pickling
    #             pickle.dump(self.data_metadata, fp)
        
    #     print("Dataset Size: " + str(len(self.data_metadata)))
        
    def __getraw__(self, index):
        # print(index)
        sparse = depth_read(self.paths['d'][index].rstrip('\n')) if \
            self.paths['d'][index] is not None else None
        target = depth_read(self.paths['gt'][index].rstrip('\n')) if \
            self.paths['gt'][index] is not None else None
        path = self.paths['d'][index].rstrip('\n') if \
            self.paths['d'][index] is not None else None
        # mask, sparse, target = self.transform(mask, sparse, target, self.opt)
        # points, _ = depth_to_pts(sparse, self.K)
        # target, _ = depth_to_pts(target, self.K)
        # # mask = np.squeeze(mask.reshape(-1,1), -1)
        # mask = mask.reshape(-1,1)
        # mask = np.delete(mask,idx,0)  # gt
        
        return sparse, target, path
        # return points, mask
        
    def __getitem__(self, index):
        sparse, target, path = self.__getraw__(index)
        sparse, target = self.transform(sparse, target, self.opt)
        points, idx = depth_to_pts(sparse, self.K)
        # points = high_to_low_res(points, 0.42)
        target, _ = depth_to_pts(target, self.K)
        idx = target[:,2]>80
        target = np.delete(target, idx, 0)
        
        if self.split == 'train':
            if self.npoints < len(points):
                pts_depth = points[:, 2]
                pts_near_flag = pts_depth < 40.0
                far_idxs_choice = np.where(pts_near_flag == 0)[0]
                near_idxs = np.where(pts_near_flag == 1)[0]
                near_idxs_choice = np.random.choice(near_idxs, self.npoints - len(far_idxs_choice), replace=False)
    
                choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                    if len(far_idxs_choice) > 0 else near_idxs_choice
                np.random.shuffle(choice)
            else:
                choice = np.arange(0, len(points), dtype=np.int32)
                np.random.shuffle(choice)
    ##---------------------------------------------------------------------------------------------##
            if len(target)>33616:
                gt_depth = target[:, 2]
                gt_near_flag = gt_depth < 40.0
                far_gt_choice = np.where(gt_near_flag == 0)[0]
                near_gt = np.where(gt_near_flag == 1)[0]
                near_gt_choice = np.random.choice(near_gt, 33616 - len(far_gt_choice), replace=False)
    
                gt_choice = np.concatenate((near_gt_choice, far_gt_choice), axis=0) \
                    if len(far_gt_choice) > 0 else near_gt_choice
                np.random.shuffle(gt_choice)
            else:
                gt_choice = np.arange(0, len(target), dtype=np.int32)
                np.random.shuffle(gt_choice)  
                assert gt_choice.shape[0]==33616, "file_name: {}".format(self.paths['gt'][index])
        
            points = points[choice, :]
            target = np.concatenate([target[gt_choice, :], points],axis = 0)
        else:
            target = np.concatenate([target, points],axis = 0)

##---------------------------------------------------------------------------------------------##
        # target = np.concatenate([target, points],axis = 0)

        return_dict = {"points":points, "target":target, "path": path}
        items = {
            key: val
            for key, val in return_dict.items() if val is not None
        }
        return items
    def __len__(self):
        return len(self.paths['d'])
        # return 100
    def collate_batch(self, batch):
        batch_size = batch.__len__()
        ans_dict = {}

        for key in batch[0].keys():
            if isinstance(batch[0][key], np.ndarray):
                ans_dict[key] = np.concatenate([batch[k][key][np.newaxis, ...] for k in range(batch_size)], axis=0)

            else:
                ans_dict[key] = [batch[k][key] for k in range(batch_size)]
                if isinstance(batch[0][key], int):
                    ans_dict[key] = np.array(ans_dict[key], dtype=np.int32)
                elif isinstance(batch[0][key], float):
                    ans_dict[key] = np.array(ans_dict[key], dtype=np.float32)

        return ans_dict