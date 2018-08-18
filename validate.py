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

# Model
parser.add_argument('--model_name', type=str, default='my_model',
                    help='perceptual model name')
parser.add_argument('--pretrained_model', type=str, default='deep',
                    help='pretrained (l1 loss) model name')
parser.add_argument('--num_channels', type=int, default=256)
parser.add_argument('--num_blocks', type=int, default=32)
parser.add_argument('--res_scale', type=float, default=0.1)
#
parser.add_argument('--epoch', type=int, default=-1,
                    help='epoch to test, default value (-1) means testing all epochs')
parser.add_argument('--num_epochs', type=int, default=200,
                    help='number of trained epochs')
parser.add_argument('--alpha', type=float, default=1,
                    help='PSNR-perceptual tradeoff')



parser.add_argument('--save_path', type=str, default='results/PIRM')
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
            print("%20s: %s" %(str(arg), str(getattr(args, arg))))
print('')

def x8_forward(img, model):
    def _transform(v, op):
        v2np = v.data.cpu().numpy()
        if op == 'vflip':
            tfnp = v2np[:, :, :, ::-1].copy()
        elif op == 'hflip':
            tfnp = v2np[:, :, ::-1, :].copy()
        elif op == 'transpose':
            tfnp = v2np.transpose((0, 1, 3, 2)).copy()
        
        ret = torch.Tensor(tfnp).cuda()

        return Variable(ret)

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
    epochs = [169, 186, 196, 200, 184, 181, 170, 189, 198, 128, 132, 164, 188, 167, 101]
    max_val = False
    phase = 'self_valid'
    if phase == 'test':
        test_path = 'data/original_data/test/benchmark/PIRM_TEST/LR'
    elif phase == 'valid':
        test_path = os.path.join('data/original_data/test/benchmark/PIRM_VAL/LR')
    elif phase == 'self_valid':
        test_path = os.path.join('data/original_data/test/benchmark/PIRM_SELF_VAL/LR')
    
    # find good common
    test_path = os.path.join('code_team_AIM/data/dataset_eval/LR_dataset/Common')

    lr_paths = glob.glob(os.path.join(test_path, '*.png'))
    lr_paths.sort()
    opt = {'num_channels': args.num_channels, 
           'depth': args.num_blocks, 
           'res_scale': args.res_scale}

    #=============Model===================
    model = Generator(opt).cuda()
    print("Number of parameters: ", sum([param.nelement() for param in model.parameters()]))
    
    if args.alpha != 1:
        model_psnr = Generator(opt).cuda()
        model_psnr_path = os.path.join('check_point/pretrain/', '{}_c{}_b{}'.format(args.pretrained_model, args.num_channels, args.num_blocks), 'best_model.pt')
        model_psnr.load_state_dict(torch.load(model_psnr_path))
    cudnn.benchmark = True

    with torch.no_grad():
        for i in range(1, 201):
            #epoch = epochs[i]
            epoch = i
            
            running_time = 0
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
               
                since = time.time()

                out = model(inp)
                if args.alpha != 1:
                    #out_psnr = model(inp)
                    out_psnr = x8_forward(inp, model_psnr)
                    out = args.alpha*out + (1 - args.alpha)*out_psnr

                e_time = time.time() - since
                print('Tested %d img(s). Time %.2fs' %(i, e_time))
                running_time += e_time
 
                out = out.data.cpu().numpy()
                out = out[0, :, :, :]
                out = out.clip(0, 255).round().astype(np.uint8)
                out = out.transpose(1, 2 , 0)

                scipy.misc.imsave(os.path.join(save_path, os.path.basename(lr_path)), out)

            avr_time = running_time/len(lr_paths)
            print('Tested epoch %d. Average time: %.2fs' %(epoch, avr_time))
            if args.epoch != -1: break # test for specified epoch
        print('Finish') 

if __name__ == '__main__':
   main() 


