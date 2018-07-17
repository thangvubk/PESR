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
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from functools import reduce
parser = argparse.ArgumentParser(description='SR benchmark')
parser.add_argument('--dataset', type=str, default='PIRM',
                    help='test dataset')
parser.add_argument('--model_name', type=str, default='my_model')
parser.add_argument('--save_path', type=str, default='results/PIRM')
parser.add_argument('--epoch', type=int, default=-1,
                    help='epoch to test, default value (-1) means testing all epochs')
parser.add_argument('--num_epochs', type=int, default=200,
                    help='number of trained epochs')

parser.add_argument('--num_channels', metavar='C', type=int, default=64)
parser.add_argument('--num_blocks', metavar='N', type=int, default = 16)
parser.add_argument('--res_scale', metavar='R', type=float, default=1)
parser.add_argument('--num_imgs', type=int, default='-1',
                    help='number of img to test, default value (-1) means testing all images')
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

    output = reduce((lambda x, y: x + y), outputlist) / len(outputlist)

    return output

def main():

    #================Data==============
    test_path = os.path.join('data/original_data/test/benchmark/PIRM_VAL/LR')
    #test_path = os.path.join('data/original_data/test/benchmark/PIRM_SELF_VAL/LR')
    lr_paths = glob.glob(os.path.join(test_path, '*.png'))
    lr_paths.sort()
    opt = {'num_channels': args.num_channels, 
           'depth': args.num_blocks, 
           'res_scale': args.res_scale}

    #=============Model===================
    model = Generator(opt).cuda()
    
    model_psnr = Generator(opt).cuda()
    #model_psnr_path = os.path.join('check_point/pretrain/', '{}/c{}_d{}'.format(args.load, args.num_channels, args.num_blocks), 'best_model.pt')
    #model_psnr.load_state_dict(torch.load(model_psnr_path))
    cudnn.benchmark = True

    for epoch in range(151, 201):
        if args.epoch != -1: epoch = args.epoch

        save_path = os.path.join(args.save_path, str(epoch))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        check_point = os.path.join('check_point2', args.model_name, str(epoch)+'.pt')
        model.load_state_dict(torch.load(check_point))
        
        for i, lr_path in enumerate(lr_paths):
            if i == args.num_imgs: break

            inp = scipy.misc.imread(lr_path)
            inp = inp.transpose(2, 0, 1)
            inp = inp[np.newaxis, :, :, :]
            inp = Variable(torch.Tensor(inp.astype(float)).cuda())
            out = model(inp)
            #out = x8_forward(inp, model) 
            #print(time.time() - since)
            
            out = out.data.cpu().numpy()
            out = out[0, :, :, :]
            out = out.clip(0, 255).round()
            out = out.transpose(1, 2 , 0)

            scipy.misc.imsave(os.path.join(save_path, os.path.basename(lr_path)), out)

        print('Tested epoch {}'.format(epoch))
        if args.epoch != -1: break
    print('Finish') 

if __name__ == '__main__':
   main() 


