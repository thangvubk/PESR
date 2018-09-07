from __future__ import division
import math
import torch
import os
import shutil
import numpy as np
from torch.autograd import Variable
import pdb

def rgb2y(rgb):
    return np.dot(rgb[...,:3], [65.738/256, 129.057/256, 25.064/256]) + 16

def tensors_to_imgs(x):
    for i in range(len(x)):
        x[i] = x[i].squeeze(0).data.cpu().numpy()
        x[i] = x[i].clip(0, 255).round()
        x[i] = x[i].transpose(1, 2, 0).astype(np.uint8)
    return x

def imgs_to_tensors(x):
    for i in range(len(x)):
        x[i] = x[i].transpose(2, 0, 1)
        x[i] = np.expand_dims(x[i], axis=0)
        x[i] = Variable(torch.Tensor(x[i].astype(float)).cuda())
    return x

def normalize(x):
    for i in range(len(x)):
        x[i] = x[i].clamp(0, 255)/255
    return x 

def compute_PSNR(out, lbl):
    [out, lbl] = tensors_to_imgs([out, lbl])
    out = rgb2y(out)
    lbl = rgb2y(lbl)
    out = out.clip(0, 255).round()
    lbl = lbl.clip(0, 255).round()
    diff = out - lbl
    rmse = np.sqrt(np.mean(diff**2))
    psnr = 20*np.log10(255/rmse)
    return psnr


def update_tensorboard(epoch, tb, img_idx, inp, out, lbl):
    [inp, out, lbl] = normalize([inp, out, lbl])
    if epoch == 1:
        tb.add_image(str(img_idx) + '_LR', inp, epoch)
        tb.add_image(str(img_idx) + '_HR', lbl, epoch)
    tb.add_image(str(img_idx) + '_SR', out, epoch)

    
def clean_and_mk_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
