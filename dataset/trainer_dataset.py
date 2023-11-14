import torch
import dataset.dataset_PUGAN as dataset_PUGAN
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

        self.datasets.dataset_train = dataset_PUGAN.PUGAN_Dataset(self.opt, 'train')
        
        
        # print("run_single_eval1:", self.opt.run_single_eval)
        if self.opt.run_single_eval:
            self.datasets.dataset_test = dataset_PUGAN.PUGAN_Dataset(self.opt, 'val')
            self.datasets.dataloader_test = torch.utils.data.DataLoader(self.datasets.dataset_test,
                                                                    batch_size=self.opt.batch_size_test,
                                                                    shuffle=False, num_workers=int(self.opt.workers))

            
        else:
            self.datasets.dataset_test = dataset_PUGAN.PUGAN_Dataset(self.opt, 'val')
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
