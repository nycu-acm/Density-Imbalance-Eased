import torch
# import dataset.dataset_KITTI_3d as dataset_KITTI
import dataset.dataset_carla_with_seg as dataset_carla
# import dataset.dataset_carla as dataset_carla
# import dataset.dataset_KITTI as dataset_KITTI
# import dataset.dataset_pu1k as dataset_pu1k
import dataset.dataset_CARLA as dataset_CARLA
import dataset.augmenter as augmenter
from easydict import EasyDict


class TrainerDataset(object):
    def __init__(self):
        super(TrainerDataset, self).__init__()

    def build_dataset(self):
        """
        Create dataset
        Author : Thibault Groueix 01.11.2019
        """

        self.datasets = EasyDict()
        # Create Datasets

        # self.datasets.dataset_train = dataset_KITTI.KittiDataset(self.opt, 'train')
        # self.datasets.dataset_test = dataset_KITTI.KittiDataset(self.opt, 'val')
        # self.datasets.dataset_train = dataset_KITTI.KittiDataset('../PointRCNN/data', 'TRAIN')
        # self.datasets.dataset_test = dataset_KITTI.KittiDataset('../PointRCNN/data', 'EVAL')
        # self.datasets.dataset_train = dataset_pu1k.PU1K_Dataset(self.opt, 'train')
        self.datasets.dataset_train = dataset_CARLA.CARLA_Dataset(self.opt, 'train')
        
        
        # print("run_single_eval1:", self.opt.run_single_eval)
        if self.opt.run_single_eval:
            # self.datasets.dataset_test = dataset_KITTI.KittiDataset(self.opt, 'val')
            self.datasets.dataset_test = dataset_carla.KittiDataset('../PointRCNN/data', 'EVAL')
            self.datasets.dataloader_test = torch.utils.data.DataLoader(self.datasets.dataset_test,
                                                                        batch_size=self.opt.batch_size_test,
                                                                        shuffle=False, num_workers=int(self.opt.workers),
                                                                        collate_fn=self.datasets.dataset_test.collate_batch)
            #############################################################################
            # self.datasets.dataset_test = dataset_pu1k.PU1K_Dataset(self.opt, 'val')
            # self.datasets.dataset_test = dataset_PUGAN.PUGAN_Dataset(self.opt, 'val')
            # self.datasets.dataloader_test = torch.utils.data.DataLoader(self.datasets.dataset_test,
            #                                                         batch_size=self.opt.batch_size_test,
            #                                                         shuffle=False, num_workers=int(self.opt.workers))
            #############################################################################

            
        else:
            # self.datasets.dataset_test = dataset_pu1k.PU1K_Dataset(self.opt, 'val')
            self.datasets.dataset_test = dataset_CARLA.CARLA_Dataset(self.opt, 'val')
            self.datasets.dataloader_test = torch.utils.data.DataLoader(self.datasets.dataset_test,
                                                                    batch_size=self.opt.batch_size_test,
                                                                    shuffle=False, num_workers=int(self.opt.workers))

        # Create dataloaders
        self.datasets.dataloader_train = torch.utils.data.DataLoader(self.datasets.dataset_train,
                                                                     batch_size=self.opt.batch_size,
                                                                     shuffle=True,
                                                                     num_workers=int(self.opt.workers))
        


        self.datasets.len_dataset = len(self.datasets.dataset_train)
        self.datasets.len_dataset_test = len(self.datasets.dataset_test)
