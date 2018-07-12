from __future__ import division
import math
import torch
import os
import shutil
import pytorch_ssim
import numpy as np
from torch.autograd import Variable

TRAIN_MEAN = np.array([114.74472479, 111.73238968, 103.34563903])
VAL_MEAN = np.array([111.44175713, 122.75937877, 101.83520272])

def rgb2y(rgb):
    return np.dot(rgb[...,:3], [65.738/256, 129.057/256, 25.064/256]) + 16

def compute_PSNR1(imgs1, imgs2):
    """imgs1 and imgs2 are torch variable of shape (N, C, H, W)"""
    print(imgs1.shape, imgs2.shape)
    N = imgs1.size()[0]
    imdiff = imgs1 - imgs2
    imdiff = imdiff.view(N, -1)
    print(imdiff.shape)
    rmse = torch.sqrt(torch.mean(imdiff**2, dim=1))
    psnr = 20*torch.log(255/rmse)/math.log(10) # psnr = 20*log10(255/rmse)
    print(psnr.shape)
    psnr =  torch.sum(psnr)
    print(psnr)
    return psnr

def compute_PSNR(out, lbl):
    out = rgb2y(out)
    lbl = rgb2y(lbl)
    diff = out - lbl
    rmse = np.sqrt(np.mean(diff**2))
    psnr = 20*np.log10(255/rmse)
    return psnr

def compute_SSIM(out, lbl):
    out = out/255
    lbl = lbl/255
    out = torch.Tensor(out[np.newaxis, np.newaxis, :, :])
    lbl = torch.Tensor(lbl[np.newaxis, np.newaxis, :, :])
    out, lbl = Variable(out.cuda()), Variable(lbl.cuda())
    return pytorch_ssim.ssim(out, lbl)

def to_numpy(epoch, tb, img_idx, inp, out, lbl):
    # update tensorboard and
    inp = inp.squeeze(0).data.cpu().numpy()
    out = out.squeeze(0).data.cpu().numpy()
    lbl = lbl.squeeze(0).data.cpu().numpy()

    out = out.clip(0, 255).round()

    inp = inp.transpose(1, 2, 0).astype(np.uint8)
    out = out.transpose(1, 2, 0).astype(np.uint8)
    lbl = lbl.transpose(1, 2, 0).astype(np.uint8)
    return inp, out, lbl

def update_tensorboard(epoch, tb, img_idx, inp, out, lbl)
    if epoch == 0:
        tb.add_image(str(img_idx) + '_LR', inp, epoch)
        tb.add_image(str(img_idx) + '_HR', lbl, epoch)
    tb.add_image(str(img_idx) + '_SR', out, epoch)

    
def clean_and_mk_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
