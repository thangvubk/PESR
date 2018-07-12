from __future__ import division
import torch
import os
import h5py
import glob
import math
import numpy as np
from torch.utils.data import Dataset
import scipy.misc
import random
import utils
import time

class DIV2K_Dataset(Dataset):
    def __init__(self, path, patch_size, scale, num_repeats, is_aug=False, crop_type=None, fixed_length=None):
        self.is_aug = is_aug
        self.crop_type = crop_type
        self.scale = scale
        str_scale = 'X' + str(scale)
        self.patch_size = patch_size
        self.num_repeats = num_repeats
        self.fixed_length = fixed_length
        random.seed(1)

        np_inputs = os.path.join(path, str_scale + '_inputs.npy')
        np_labels = os.path.join(path, 'labels.npy')

        if os.path.exists(np_inputs) and os.path.exists(np_labels):
            self.inputs = np.load(np_inputs)
            self.labels = np.load(np_labels)
        else:
            print('Cannot find numpy file. Reading image...')
            since = time.time()
            hr_path = os.path.join(path, 'HR')
            lr_path = os.path.join(path, 'LR', str_scale)
            hr_globs = glob.glob(os.path.join(hr_path, '*.png'))
            lr_globs = glob.glob(os.path.join(lr_path, '*.png'))
            hr_globs.sort()
            lr_globs.sort()
            self.inputs = [scipy.misc.imread(inp) for inp in lr_globs]
            self.labels = [scipy.misc.imread(lbl) for lbl in hr_globs]
            print('Complete reading images in %f seconds' %(time.time() - since))

            print('Writing data to npy...')
            since = time.time()
            np.save(os.path.join(path, str_scale + '_inputs.npy'), self.inputs)
            np.save(os.path.join(path, 'labels.npy'), self.labels)
            print('Complete writing in %f seconds' %(time.time() - since))
       
        # simple test
        #self.inputs = self.inputs[0:1024]
        #self.labels = self.labels[0:1024]
    def __len__(self):
        if self.fixed_length is not None:
            return self.fixed_length
        return len(self.inputs)*self.num_repeats
    
    def __getitem__(self, idx):
        idx = idx % len(self.inputs)

        inp = self.inputs[idx]#.astype(np.float32).copy()
        lbl = self.labels[idx]#.astype(np.float32).copy()

        #inp, lbl = self._normalize(inp, lbl)

        if self.crop_type is not None:
            inp, lbl = self._crop(inp, lbl, self.crop_type)
            
        if self.is_aug:
            inp, lbl = self._aug_data(inp, lbl)

        inp, lbl = self._to_tensor(inp, lbl)
        return inp, lbl

    def _aug_data(self, inp, lbl):
        
        aug_idx = random.randint(0,7)
        assert aug_idx >= 0
        assert aug_idx <= 7

        if (aug_idx>>2)&1 == 1:
            # transpose
            inp = inp.transpose((1, 0, 2)).copy()
            lbl = lbl.transpose((1, 0, 2)).copy()
        if (aug_idx>>1)&1 == 1:
            # vertical flip
            inp = inp[::-1, :, :].copy()
            lbl = lbl[::-1, :, :].copy()
        if aug_idx&1 == 1:
            # horizontal flip
            inp = inp[:, ::-1, :].copy()
            lbl = lbl[:, ::-1, :].copy()

        return inp, lbl
    
    def _crop(self, inp, lbl, crop_type):
        ih, iw, ic = inp.shape #shape of original image

        inp_patch_size = self.patch_size
        lbl_patch_size = inp_patch_size*self.scale

        if crop_type is 'random':
            # indexing inp patch
            h = random.randint(0, ih - inp_patch_size)
            w = random.randint(0, iw - inp_patch_size)
            # indexing lbl patch
            H = h*self.scale
            W = w*self.scale
        elif crop_type is 'fixed':
            h, w, H, W = 0, 0, 0, 0
        else:
            raise Exception('Unknown crop type: {}'.format(crop_type))

        inp = inp[h:h+inp_patch_size, w:w+inp_patch_size, :]
        lbl = lbl[H:H+lbl_patch_size, W:W+lbl_patch_size, :]

        return inp, lbl

    def _to_tensor(self, inp, lbl):
        inp = inp.transpose(2, 0, 1)
        lbl = lbl.transpose(2, 0, 1)
        return torch.FloatTensor(inp), torch.FloatTensor(lbl)

    def _normalize(self, inp, lbl):
        # transpose to channel-last image
        inp = inp.transpose(1, 2, 0)
        lbl = lbl.transpose(1, 2, 0)
        inp = (inp - self.channel_means)/255
        lbl = (lbl - self.channel_means)/255
        inp = inp.transpose(2, 0, 1)
        lbl = lbl.transpose(2, 0, 1)
        return inp, lbl
        
        
        
        
