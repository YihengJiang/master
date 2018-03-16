import scipy.io as spio
import numpy as np
import torch.utils.data as data
import torch

def default_loader(path):
    return spio.loadmat(path)['out']

class getBatchTrn(data.Dataset):
    def __init__(self, utterlist, rois, roislabels, nduration, ndim, data_root,
                 transform=None, target_transform=None, loader=default_loader):
        self.utterlist = utterlist
        self.rois = rois
        self.loader = loader
        self.labels = (roislabels)
        self.nduration = int(nduration)
        self.ndim = int(ndim)
        self.data_root = data_root
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, indx):
        utterpath = self.utterlist[self.rois[indx,0]]
        utter = self.loader(utterpath)
        st_ = self.rois[indx, 1]
        ed_ = self.rois[indx, 2]
        length_roi = ed_ - st_
        assert (length_roi <= self.nduration)
        utter_roi = np.zeros(shape=(1, self.nduration, self.ndim), dtype=float)
        utter_roi[:, 0:length_roi, :] = utter[:,st_:ed_, :]

        utter_roi= torch.from_numpy(utter_roi.transpose((2, 0, 1)))
        utter_roi = utter_roi.float()
        target_roi = int(self.labels[indx])

        return utter_roi, target_roi

    def __len__(self):
        return min(len(self.rois),1e11)


class getBatchVal(data.Dataset):
    def __init__(self, utterlist, labels, data_root, transform=None, target_transform=None, loader=default_loader):
        self.utterlist = utterlist
        self.loader = loader
        self.labels = labels
        self.data_root = data_root
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, indx):
        utterpath = self.utterlist[indx]
        #print utterpath
        utter  = self.loader(utterpath)
        #print utter.shape
        target = int(self.labels[indx])
        h, w, c = utter.shape
        if w < 30:
            utter_roi = np.zeros(shape=(h, 30, c), dtype=float)
            utter_roi[:, 0:w, :] = utter
            utter = utter_roi
        utter= torch.from_numpy(utter.transpose((2, 0, 1)))
        utter = utter.float()

        return utter, target


    def __len__(self):
        return len(self.utterlist)





