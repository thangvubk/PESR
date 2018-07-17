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
parser.add_argument('-p', '--check_point', metavar='T', type=str, default='best_model')
parser.add_argument('--save_path', type=str, default='results/PIRM')
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
    val_set_path = os.path.join('data/original_data/test/benchmark', args.dataset)
    val_set = SRDataset(val_set_path, patch_size=None, num_repeats=1, scale=args.scale, is_aug=False)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=8)
    test_path = os.path.join('data/preprocessed_data/test', args.dataset, args.str_scale)
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    lr_paths = glob.glob(os.path.join(test_path, 'low_res', '*.bmp'))
    hr_paths = glob.glob(os.path.join(test_path, 'high_res', '*.bmp'))    
    lr_paths += glob.glob(os.path.join(test_path, 'low_res', '*.jpg'))
    hr_paths += glob.glob(os.path.join(test_path, 'high_res', '*.jpg'))
    lr_paths += glob.glob(os.path.join(test_path, 'low_res', '*.png'))
    hr_paths += glob.glob(os.path.join(test_path, 'high_res', '*.png')) 
    lr_paths.sort()
    hr_paths.sort()
    
    check_point = args.check_point
    if not os.path.exists(check_point):
        raise Exception('Cannot find %s' %check_point)

    opt = {'scale': args.scale, 'num_channels': args.num_channels, 'depth': args.num_blocks, 'res_scale': args.res_scale}
    model = Generator(opt).cuda()
    model.load_state_dict(torch.load(check_point))

    # test psnr
    model_psnr = Generator(opt).cuda()
    model_psnr.load_state_dict(torch.load('check_point/pretrain/ReLU_DIV2K900_300epoch_256feats/c256_d32/best_model.pt'))

    psnrs = []
    ssims = []
    for i, (inp, lbl) in enumerate(val_loader):
        if i == args.num_imgs: break
        inp = Variable(inp.cuda())
        out = model(inp)

        out_psnr = model_psnr(inp)
        alpha = 0
        out = alpha*out + (1-alpha)*out_psnr

        #out = x8_forward(inp, model) 
        #print(time.time() - since)
        
        #out = (out)*255 + mean
        out = out.data.cpu().numpy()
        out = out[0, :, :, :]
        out = out.clip(0, 255).round()
        out = out.transpose(1, 2 , 0)
        #out = utils.rgb2y(out)
        #out = out.astype(np.float32)

        lbl = lbl.numpy()
        lbl = lbl[0, :, :, :]
        lbl = lbl.transpose(1, 2, 0)
        #lbl = utils.rgb2y(lbl)
        #lbl = lbl.astype(np.float32)



        #(hei, wid) = out.shape
        #out = out[4:hei-4, 4:wid-4]
        #lbl = lbl[4:hei-4, 4:wid-4]

        #psnr = utils.compute_PSNR(out, lbl)
        #ssim = utils.compute_SSIM(out, lbl)
        #psnrs.append(psnr)
        #ssims.append(ssim)
        #print(psnr)

        #print('%20s: %.3fdB %.4f' %(os.path.basename(lr_path), psnr, ssim))
        scipy.misc.imsave(os.path.join(save_path, '{}.png'.format(i+1)), out)
 
    print('average: %.4fdB %.4f' %(np.mean(psnrs), np.mean(ssims)))

if __name__ == '__main__':
   main() 


