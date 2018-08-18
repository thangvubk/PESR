from __future__ import division
import scipy.misc
import glob
import numpy as np
from model import *
from utils import *
import time, os, argparse
from torch.autograd import Variable
from data import *
import torch.backends.cudnn as cudnn
from functools import reduce


parser = argparse.ArgumentParser(description='SR benchmark')

#dataset
parser.add_argument('--dataset', type=str, default='PIRM',
                    help='test dataset')

# Model
parser.add_argument('--perceptual_model', type=str, default='check_point',
                    help='perceptual model name')
parser.add_argument('--psnr_model', type=str, default='deep',
                    help='pretrained (l1 loss) model name')
parser.add_argument('--num_channels', type=int, default=256)
parser.add_argument('--num_blocks', type=int, default=32)
parser.add_argument('--res_scale', type=float, default=0.1)

# perceptual degree
parser.add_argument('--alpha', type=float, default=1,
                    help='PSNR-perceptual tradeoff')

parser.add_argument('--save_path', type=str, default='results')
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
    path = os.path.join('data/origin/test/', args.dataset, 'LR')
    lr_paths = glob.glob(os.path.join(path, '*.png'))

    #=============Model===================
    opt = {'num_channels': args.num_channels, 
           'depth': args.num_blocks, 
           'res_scale': args.res_scale}
    model = Generator(opt).cuda()
    model.load_state_dict(torch.load(args.perceptual_model))
    print("Number of parameters: ", sum([param.nelement() for param in model.parameters()]))
    
    if args.alpha != 1:
        model_psnr = Generator(opt).cuda()
        model_psnr.load_state_dict(torch.load(psnr_model))

    cudnn.benchmark = True

    save_path = os.path.join(args.save_path, args.dataset)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with torch.no_grad():
        for i, lr_path in enumerate(lr_paths):
            inp = scipy.misc.imread(lr_path)
            [inp] = imgs_to_tensors([inp])

            out = model(inp)
            if args.alpha != 1:
                out_psnr = x8_forward(inp, model_psnr)
                out = args.alpha*out + (1 - args.alpha)*out_psnr

            print('Tested %d img(s)' %(i+1))
            [out] = tensors_to_imgs([out])

            scipy.misc.imsave(os.path.join(save_path, os.path.basename(lr_path)), out)

        print('Finish') 

if __name__ == '__main__':
   main() 


