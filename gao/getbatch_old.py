import scipy.io as spio
import numpy as np
import torch.utils.data as data
import torch
import random

def default_loader(path):
    return spio.loadmat(path)['out']

class getBatchTrn(data.Dataset):
    def __init__(self, egs, ndim, data_root,egs_num,
                 transform=None, target_transform=None, loader=default_loader):
        self.egs_all = egs
        self.loader = loader
        self.ndim = int(ndim)
        self.data_root = data_root
        self.transform = transform
        self.target_transform = target_transform
        self.epoch = 0
        self.it = 0
        self.egs_num = egs_num

    def __getitem__(self, indx):
        egs_id = self.it % self.egs_num
        egs = self.egs_all[egs_id]
        eg = egs[indx]
        utterpath = eg[2]
        utter = self.loader(utterpath)
        utter = utter.reshape((1, utter.shape[0], utter.shape[1]))
        utter_roi = torch.from_numpy(utter.transpose((0, 2, 1))).float()
        # utter_roi = utter_roi.float()
        target_roi = int(eg[0])
        target_spkid = eg[1]
        return utter_roi, target_roi

    def __len__(self):
        len_min_egs = min(len(x) for x in self.egs_all)
        return min(len_min_egs,1e11)

class getBatchVal(data.Dataset):
    def __init__(self, egs, ndim, data_root, egs_num,
                 transform=None, target_transform=None, loader=default_loader):
        self.egs_all = egs
        self.loader = loader
        self.ndim = int(ndim)
        self.data_root = data_root
        self.transform = transform
        self.target_transform = target_transform
        self.epoch = 0
        self.it = 0
        self.egs_num = egs_num

    def __getitem__(self, indx):
        egs_id = 0
        egs = self.egs_all[egs_id]
        eg = egs[indx]
        utterpath = eg[2]
        utter = self.loader(utterpath)
        utter = utter.reshape((1, utter.shape[0], utter.shape[1]))
        utter_roi = torch.from_numpy(utter.transpose((0, 2, 1))).float()
        # utter_roi = utter_roi.float()
        target_roi = int(eg[0])
        target_spkid = eg[1]
        return utter_roi, target_roi

    def __len__(self):
        return min(len(self.egs_all[0]), 1e11)





