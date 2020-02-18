import os
import math

import re

import sys
sys.path.append('./)
import transform_2d as trans

from skimage import io, color
import numpy as np
from torch.utils.data import Dataset




class BaseData(Dataset):

    def __init__(self, transforms=None):

        self.to_tensor = trans.ToTensor()
        self.normalize = trans.Normalize()

        ## self.all_data = np.load(data_file, mmap_mode='r')
        self.to_tensor = trans.ToTensor()
        self.normalize = trans.Normalize()

        self.basenames = []

        assert isinstance(transforms, list) or (not transforms)
        if transforms == None:
            transforms = []
        self.transforms = transforms

    def __len__(self):
        return len(self.basenames)

    def read_label(self, path):
        lb = io.imread(path)
        if len(lb.shape) == 3:
            if lb.shape[2] == 4:
                lb = lb[:, :, 3]
            elif lb.shape[2] == 3:
                lb = color.rgb2gray(lb)
        return lb
    
    ## TODO: make it more general
    def make_transforms(self, data, data_channels, lbs=[]):
        for tr in self.transforms:
            assert hasattr(tr, 'target')
            if tr.target == 'both' and len(lbs) > 0:
                sample = np.concatenate((data, lbs), axis=2)
                sample = tr(sample)
                data, lbs = sample[:, :, :data_channels], sample[:, :, data_channels:]
            elif (tr.target == 'both' and len(lbs) == 0) or tr.target == 'data':
                data = tr(data)
            elif tr.target == 'labels' and len(lbs) > 0:
                lbs = tr(lbs)
        
        return data, lbs


class DataProstate(BaseData):
    """
        Dataset for Prostate Seg
    """

    def __init__(self, path='./', transforms=None, preload=False):
        super().__init__(transforms)

        self.path = path
        im_names = os.listdir(path)
        self.basenames = []

        for n in im_names:
            p = os.path.join(path, n)
            if os.path.isdir(p):
                self.basenames.append(p)
        
        # if ids: self.basenames = [self.basenames[idx] for idx in ids]
    
    def cartesian_position(self, dim_x, dim_y):  # shape[0], shape[1]
        coord_y = np.repeat(np.expand_dims(np.linspace(
            0, 1, dim_y), axis=1), dim_x, axis=1)
        coord_x = np.repeat(
            [np.linspace(0, 1, dim_x)], dim_y, axis=0)
        return coord_y, coord_x

    def __len__(self):
        return len(self.basenames)

    def __getitem__(self, idx):
        # if len(self.imgs) > 0: data = self.imgs[idx]
        basepath = self.basenames[idx]

        data = io.imread(os.path.join(basepath, 'CT.jpg'))
        dim_y, dim_x = data.shape[:2]
        if data.ndim == 2: data = np.expand_dims(data, axis=2)

        lbnames = ['Prostate.jpg', 'Bladder.jpg', 'PenileBulb.jpg', 'Rectum.jpg']
        lb_num = len(lbnames)

        lbs = [(io.imread(os.path.join(basepath, n)) > 128).astype('float32') for n in lbnames]
        lbs += self.cartesian_position(dim_y, dim_x) ## coordinates
        lbs += np.array(lbs).transpose(1, 2, 0)

        ## TODO: add weights to the loss
        # weight = np.ones([dim_y, dim_x, lb_num], dtype='float32')
        data, lbs = self.make_transforms(data, 3, lbs)

        lbs, coords = lbs[...,:lb_num], lbs[...,-2:]
        ## coords are the guiding pixels for 
        data = np.concatenate([data, coords], axis=2)

        sample = {
            'data': self.to_tensor(data.astype('float32')),
            # 'weights': self.to_tensor(weight.astype('float32')),
            'labels': self.to_tensor(lbs.astype('float32')),
            'img_name': os.path.basename(basepath)
            }

        return sample
