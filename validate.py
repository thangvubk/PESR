from __future__ import division
import scipy.misc
import glob
import numpy as np
from model import *
import utils
import time
import os
import argparse
from torch.autograd import Variable
import pytorch_ssim
import h5py
from data import *
from torch.utils.data import DataLoader
parser = argparse.ArgumentParser(description='SR benchmark')
parser.add_argument('-m', '--model', metavar='M', type=str, default='VDSR',
                    help='network architecture. Default SRCNN')
parser.add_argument('-s', '--scale', metavar='S', type=int, default=4, 
                    help='interpolation scale. Default 4')
parser.add_argument('-t', '--dataset', metavar='T', type=str, default='PIRM')
parser.add_argument('-p', '--check-point', metavar='T', type=str, default='best_model')
parser.add_argument('--save-path', type=str, default='results/PIRM')
#temp ssim tuning
parser.add_argument('-a', '--alpha', metavar='A', type=str, default="") 
parser.add_argument('-c', '--num_channels', metavar='C', type=int, default=64)
parser.add_argument('-d', '--num_blocks', metavar='N', type=int, default = 16)
parser.add_argument('-r', '--res_scale', metavar='R', type=float, default=1)
parser.add_argument('--num_imgs', type=int, default='-1')
args = parser.parse_args()

print('############################################################')
print('# Image Super Resolution - Pytorch implementation          #')
print('# by Thang Vu                                              #')
print('############################################################')
print('')
print('-------YOUR SETTINGS_________')
for arg in vars(args):
            print("%15s: %s" %(str(arg), str(getattr(args, arg))))
print('')
args.str_scale = 'X' + str(args.scale)

def x8_forward(img, model, precision='single'):
    def _transform(v, op):
        if precision != 'single': v = v.float()

        v2np = v.data.cpu().numpy()
        if op == 'vflip':
            tfnp = v2np[:, :, :, ::-1].copy()
        elif op == 'hflip':
            tfnp = v2np[:, :, ::-1, :].copy()
        elif op == 'transpose':
            tfnp = v2np.transpose((0, 1, 3, 2)).copy()
        
        ret = torch.Tensor(tfnp).cuda()

        if precision == 'half':
            ret = ret.half()
        elif precision == 'double':
            ret = ret.double()

        return Variable(ret, volatile=v.volatile)

    inputlist = [img]
    for tf in 'vflip', 'hflip', 'transpose':
        inputlist.extend([_transform(t, tf) for t in inputlist])

    outputlist = [model(aug) for aug in inputlist]
    for i in range(len(outputlist)):
        if i > 3:
            outputlist[i] = _transform(outputlist[i], 'transpose')
        if i % 4 > 1:
            outputlist[i] = _transform(outputlist[i], 'hflip')
        if (i % 4) % 2 == 1:
            outputlist[i] = _transform(outputlist[i], 'vflip')
    #print(len(outputlist))
    output = reduce((lambda x, y: x + y), outputlist) / len(outputlist)

    return output

def main():
    test_path = os.path.join('data/original_data/test/benchmark/PIRM_VAL/LR')
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    lr_paths = glob.glob(os.path.join(test_path, '*.png'))
    lr_paths.sort()
    
    check_point = args.check_point
    if not os.path.exists(check_point):
        raise Exception('Cannot find %s' %check_point)

    opt = {'scale': args.scale, 'num_channels': args.num_channels, 'depth': args.num_blocks, 'res_scale': args.res_scale}
    model = Generator_L2H(opt).cuda()
    model.load_state_dict(torch.load(check_point))
    model.cuda()
    for lr_path in lr_paths:
        inp = scipy.misc.imread(lr_path)
        inp = inp.transpose(2, 0, 1)
        inp = inp[np.newaxis, :, :, :]
        inp = Variable(torch.Tensor(inp.astype(float)).cuda())
        out = model(inp)
        #out = x8_forward(inp, model) 
        #print(time.time() - since)
        
        #out = (out)*255 + mean
        out = out.data.cpu().numpy()
        out = out[0, :, :, :]
        out = out.clip(0, 255).round()
        out = out.transpose(1, 2 , 0)

        scipy.misc.imsave(os.path.join(save_path, os.path.basename(lr_path)), out)
    print('Finish') 

if __name__ == '__main__':
   main() 


