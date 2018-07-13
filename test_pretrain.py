from __future__ import division
import scipy.misc
import glob
import numpy as np
from model import *
from utils import *
import time
import os
import argparse
from torch.autograd import Variable
import h5py
from data import *
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='SR benchmark')

parser.add_argument('-s', '--scale', metavar='S', type=int, default=4, 
                    help='interpolation scale. Default 4')
parser.add_argument('-t', '--dataset', metavar='T', type=str, default='Set5')
parser.add_argument('--load', type=str, default='model')

parser.add_argument('-c', '--num_channels', metavar='C', type=int, default=64)
parser.add_argument('-d', '--num_blocks', metavar='N', type=int, default = 16)
parser.add_argument('-r', '--res_scale', metavar='R', type=float, default=1)
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
    val_set_path = os.path.join('data/original_data/test/benchmark', args.dataset)
    val_set = SRDataset(val_set_path, patch_size=None, num_repeats=1, scale=args.scale, is_aug=False)
    val_loader = DataLoader(val_set, batch_size=1,
                            shuffle=False, num_workers=8)
    
    check_point = os.path.join('check_point/pretrain', '{}/c{}_d{}'.format(args.load, args.num_channels, args.num_blocks), 'best_model.pt')
    if not os.path.exists(check_point):
        raise Exception('Cannot find %s' %check_point)

    opt = {'scale': args.scale, 'num_channels': args.num_channels, 'depth': args.num_blocks, 'res_scale': args.res_scale}
    model = Generator(opt)
    model.load_state_dict(torch.load(check_point))
    model.cuda()
    psnrs = []
    with torch.no_grad():
        for i, (inp, lbl) in enumerate(val_loader):
            inp, lbl = (Variable(inp.cuda()),
                        Variable(lbl.cuda()))
            out = model(inp)
            inp, out, lbl = to_numpy(inp, out, lbl)

            (hei, wid, _) = out.shape
            out = out[4:hei-4, 4:wid-4]
            lbl = lbl[4:hei-4, 4:wid-4]

            psnr = utils.compute_PSNR(out, lbl)
            psnrs.append(psnr)
            print(i, psnr)
 
    print('average: %.4fdB' %np.mean(psnrs))

if __name__ == '__main__':
   main() 

