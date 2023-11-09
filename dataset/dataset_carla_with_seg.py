import os
import numpy as np
import torch.utils.data as torch_data
import kitti_util
import cv2
from PIL import Image


USE_INTENSITY = False


class KittiDataset(torch_data.Dataset):
    def __init__(self, root_dir, split='train',mode = 'TRAIN'):
        self.split = split
        self.mode = mode
        self.classes = ['Car']
        is_test = self.split == 'test'
        self.imageset_dir = os.path.join(root_dir, 'KITTI', 'object_carla', 'testing' if is_test else 'training')
        # print(self.mode)
        
        if self.split =='train':
            split_dir = os.path.join(root_dir, 'KITTI', 'object_carla', 'training', 'train.txt')
        else:
            # print("True")
            split_dir = os.path.join(root_dir, 'KITTI', 'object_carla', 'training', 'val.txt')

        self.image_idx_list = [x.strip() for x in open(split_dir).readlines()]
        self.sample_id_list = [int(sample_id) for sample_id in self.image_idx_list]
        self.num_sample = self.image_idx_list.__len__()

        self.npoints = 16384

        self.image_dir = os.path.join(self.imageset_dir, 'image_2')
        self.lidar_dir = os.path.join(self.imageset_dir, 'velodyne')
        self.calib_dir = os.path.join(self.imageset_dir, 'calib')
        self.label_dir = os.path.join(self.imageset_dir, 'label_2')
        self.plane_dir = os.path.join(self.imageset_dir, 'planes')
        self.gt_dir = os.path.join(self.imageset_dir, 'gt_points')
        # self.gt_dir = os.path.join(self.imageset_dir, 'velodyne')
        self.seg_dir = os.path.join(self.imageset_dir, 'segmentation')
    

    def get_image(self, idx):
        img_file = os.path.join(self.image_dir, '%06d.png' % idx)
        assert os.path.exists(img_file)
        return cv2.imread(img_file)  # (H, W, 3) BGR mode

    def get_image_shape(self, idx):
        img_file = os.path.join(self.image_dir, '%06d.png' % idx)
        assert os.path.exists(img_file)
        im = Image.open(img_file)
        width, height = im.size
        return height, width, 3
    
    def get_seg(self, idx):
        seg_file = os.path.join(self.seg_dir, '%06d.png' % idx)
        assert os.path.exists(seg_file)
        seg = cv2.imread(seg_file)
        seg = seg[:,:,::-1]
        return seg # (H, W, 3) BGR mode

    def get_lidar(self, idx):
        lidar_file = os.path.join(self.lidar_dir, '%06d.bin' % idx)
        # print(lidar_file)
        assert os.path.exists(lidar_file)
        return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4), lidar_file
    
    def get_gt(self, idx):
        lidar_file = os.path.join(self.gt_dir, '%06d.bin' % idx)
        # print(lidar_file)
        assert os.path.exists(lidar_file)
        return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, '%06d.txt' % idx)
        assert os.path.exists(calib_file)
        return kitti_util.Calibration(calib_file)

    def get_label(self, idx):
        label_file = os.path.join(self.label_dir, '%06d.txt' % idx)
        assert os.path.exists(label_file)
        return kitti_util.get_objects_from_label(label_file)

    def transform_seg(self, seg):
        classes = {
            0: [0, 0, 0],         # None
            1: [70, 70, 70],      # Buildings
            2: [190, 153, 153],   # Fences
            3: [72, 0, 90],       # Other
            4: [220, 20, 60],     # Pedestrians
            5: [153, 153, 153],   # Poles
            6: [157, 234, 50],    # RoadLines 11111111
            7: [128, 64, 128],    # Roads 22222222
            8: [244, 35, 232],    # Sidewalks 3333333
            9: [107, 142, 35],    # Vegetation
            10: [0, 0, 255],      # Vehicles
            11: [102, 102, 156],  # Walls
            12: [220, 220, 0]     # TrafficSigns
        }
        result = np.zeros((seg.shape[0], seg.shape[1], 1))
        for key, value in classes.items():
            # print(value)
            # idx = seg[]
            idx = seg==value
            idx = idx[:,:,0]*idx[:,:,1]*idx[:,:,2]
            # result[numpy.where(seg == value)] = key
            result[idx]=key
        return result
    def get_point_class(self, seg, valid):
        result = np.zeros((valid.shape[0]))
        for i in range(result.shape[0]):
            result[i] = seg[int(valid[i,1]), int(valid[i,0]),:]
        return result

    @staticmethod        
    def get_valid_flag(pts_rect, pts_img, pts_rect_depth, img_shape):
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)
        return pts_valid_flag

    def filtrate_objects(self, obj_list):
        type_whitelist = self.classes
        if self.mode == 'TRAIN':
            type_whitelist = list(self.classes)
            if 'Car' in self.classes:
                type_whitelist.append('Van')

        valid_obj_list = []
        for obj in obj_list:
            if obj.cls_type not in type_whitelist:
                continue

            valid_obj_list.append(obj)
        return valid_obj_list

    def __len__(self):
        print(len(self.sample_id_list))
        return len(self.sample_id_list)

    def __getitem__(self, index):
        sample_id = int(self.sample_id_list[index])
        calib = self.get_calib(sample_id)
        img_shape = self.get_image_shape(sample_id)
        pts_lidar, path = self.get_lidar(sample_id)
        pts_gt = self.get_gt(sample_id)
        seg = self.get_seg(sample_id)
        seg_mask = self.transform_seg(seg)
        # print(seg_mask.shape)

        # get valid point (projected points should be in image)
        pts_rect = calib.project_velo_to_rect(pts_lidar[:, 0:3])

        pts_img, pts_rect_depth = calib.project_rect_to_image(pts_rect)

        pts_valid_flag = self.get_valid_flag(pts_rect, pts_img, pts_rect_depth, img_shape)
        
        pts_rect = pts_rect[pts_valid_flag][:, 0:3]
        
    #############################################################################################
        pts_img_valid = pts_img[pts_valid_flag,:]
        pts_class = self.get_point_class(seg_mask, pts_img_valid)
        flag1 = pts_class!=6 
        flag2 = pts_class!=7
        flag3 = pts_class!=8 
        # flag4 = pts_class!=0
        # class_valid_flag = flag1&flag2&flag3&flag4
        class_valid_flag = flag1&flag2&flag3
        pts_rect_front = pts_rect[class_valid_flag][:, 0:3]
        pts_rect_bk = pts_rect[class_valid_flag==False][:, 0:3]
    ##############################################################################################

        gt_rect = calib.project_velo_to_rect(pts_gt[:, 0:3])

        gt_img, gt_rect_depth = calib.project_rect_to_image(gt_rect)
        gt_valid_flag = self.get_valid_flag(gt_rect, gt_img, gt_rect_depth, img_shape)

        gt_rect = gt_rect[gt_valid_flag][:, 0:3]
        
        # ret_pts_rect = pts_rect
        # ret_pts_rect = calib.project_rect_to_velo(ret_pts_rect)
        
        # ret_gt_rect = gt_rect
        # ret_gt_rect = calib.project_rect_to_velo(ret_gt_rect)
        
    #############################################################################################
        gt_img_valid = gt_img[gt_valid_flag,:]
        gt_class = self.get_point_class(seg_mask, gt_img_valid)
        flag5 = gt_class!=6 
        flag6 = gt_class!=7
        flag7 = gt_class!=8 
        # flag8 = gt_class!=0
        # gt_class_valid_flag = flag5&flag6&flag7&flag8
        gt_class_valid_flag = flag5&flag6&flag7
        gt_rect_front = gt_rect[gt_class_valid_flag][:, 0:3]
        gt_rect_bk = gt_rect[gt_class_valid_flag==False][:, 0:3]

    ##############################################################################################
        ret_gt_rect = calib.project_rect_to_velo(gt_rect_front)
        ret_pts_rect = calib.project_rect_to_velo(pts_rect_front)

        sample_info = {'sample_id': sample_id}

        sample_info['points'] = ret_pts_rect
        sample_info['target'] = ret_gt_rect
        sample_info['path'] = path
        return sample_info

    @staticmethod
    def generate_training_labels(pts_rect, gt_boxes3d):
        cls_label = np.zeros((pts_rect.shape[0]), dtype=np.int32)
        gt_corners = kitti_util.boxes3d_to_corners3d(gt_boxes3d, rotate=True)
        extend_gt_boxes3d = kitti_util.enlarge_box3d(gt_boxes3d, extra_width=0.2)
        extend_gt_corners = kitti_util.boxes3d_to_corners3d(extend_gt_boxes3d, rotate=True)
        for k in range(gt_boxes3d.shape[0]):
            box_corners = gt_corners[k]
            fg_pt_flag = kitti_util.in_hull(pts_rect, box_corners)
            cls_label[fg_pt_flag] = 1

            # enlarge the bbox3d, ignore nearby points
            extend_box_corners = extend_gt_corners[k]
            fg_enlarge_flag = kitti_util.in_hull(pts_rect, extend_box_corners)
            ignore_flag = np.logical_xor(fg_pt_flag, fg_enlarge_flag)
            cls_label[ignore_flag] = -1

        return cls_label

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
