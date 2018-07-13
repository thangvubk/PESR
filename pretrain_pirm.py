from __future__ import division
from __future__ import print_function
from data import *
from model import *
from solver import *
from utils import *
import progressbar
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import pytorch_ssim
import argparse
import os 
import scipy.misc
import time
from tqdm import tqdm
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn

parser = argparse.ArgumentParser(description='SR benchmark')

# Dataset
parser.add_argument('--train-dataset', metavar='T', type=str, default='DIV2K',
                    help='Training dataset')
parser.add_argument('--valid-dataset', metavar='T', type=str, default='DIV2K',
                    help='Training dataset')
parser.add_argument('-s', '--scale', metavar='S', type=int, default=4, 
                    help='interpolation scale. Default 4. Currently, support 4x only')
parser.add_argument('--patch-size', metavar='P', type=int, default=48,
                    help='input patch size')

# Model
parser.add_argument('-c', '--num-channels', metavar='N', type=int, default = 64)
parser.add_argument('-d', '--num-blocks', metavar='N', type=int, default = 16)
parser.add_argument('-r', '--res-scale', metavar='R', type=float, default=1)

# Training 
parser.add_argument('-b', '--batch-size', metavar='B', type=int, default=16,
                    help='batch size used for training. Default 16')
parser.add_argument('-l', '--learning-rate', metavar='L', type=float, default=1e-4,
                    help='learning rate used for training. Default 1e-4')
parser.add_argument('-n', '--num-epochs', metavar='N', type=int, default=300,
                    help='number of training epochs. Default 100')
parser.add_argument('--num-repeats', metavar='V', type=int, default=20)

# Checkpoint
parser.add_argument('--save', type=str, default='model')


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

def main(argv=None):
    # ============Dataset===============
    print('Loading dataset...')
    since = time.time()
    train_set_path = os.path.join('data/original_data/train', args.train_dataset)
    val_set_path = os.path.join('data/original_data/valid', args.valid_dataset)

    train_set = SRDataset(train_set_path, patch_size=args.patch_size, num_repeats=args.num_repeats, 
                          scale=args.scale, is_aug=True, crop_type='random')
    val_set = SRDataset(val_set_path, patch_size=None, num_repeats=1, scale=args.scale, 
                        is_aug=False, fixed_length=20)
    print('Finish loading dataset in %d seconds' %(time.time() - since))
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1,
                            shuffle=False, num_workers=4, pin_memory=True)

    #===========Model=====================
    n_GPUs = torch.cuda.device_count()
    print('Loading model using %d GPU(s)...' %n_GPUs)
    opt = {'scale': args.scale, 'num_channels': args.num_channels, 'depth': args.num_blocks, 'res_scale': args.res_scale}
    model = nn.DataParallel(Generator(opt)).cuda()
    cudnn.benchmark = True
        
    check_point = os.path.join('check_point/pretrain/', '{}/c{}_d{}'.format(args.save, args.num_channels, args.num_blocks))
    clean_and_mk_dir(check_point)

    #============optimizer
    trainable = filter(lambda x: x.requires_grad, model.parameters())
    optimizer = optim.Adam(trainable,
                           lr=args.learning_rate,
                           weight_decay=1e-6)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    l1_loss_fn = nn.L1Loss()

    # ==========Log and book-keeping vars =======
    tb = SummaryWriter(check_point)
    (best_val_psnr, best_epoch) = (-1, -1)

    # Training and validating
    for epoch in range(args.num_epochs):

        #=============Training====================
        scheduler.step()
        cur_lr = optimizer.param_groups[0]['lr']
        print('Model {}. Epoch {}. Learning rate: {}'.format(
            args.save, epoch, cur_lr))

        num_batches = len(train_set)//args.batch_size
        running_loss = 0
        for i, (inputs, labels) in enumerate(tqdm(train_loader)):
            inputs, labels = (Variable(inputs.cuda()),
                              Variable(labels.cuda()))
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = l1_loss_fn(outputs, labels) 

            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        training_loss = running_loss/num_batches
        print('Epoch %d: loss %f' %(epoch, training_loss))
        tb.add_scalar('Pretrain Loss', training_loss, epoch)

        #===============Saving model==========================
        print('Saving model at epoch %d' %epoch)
        model_path = os.path.join(check_point,  'model_{}.pt'.format(epoch))
        newest_model_path = os.path.join(check_point, 'newest_model.pt')
        if torch.cuda.device_count() > 1:
            torch.save(model.module.state_dict(), model_path)
            torch.save(model.module.state_dict(), newest_model_path)
        else:
            torch.save(model.state_dict(), model_path)
            torch.save(model.state_dict(), newest_model_path)

        #================Validating============================
        print('Validating...')
        val_psnr = 0
        with torch.no_grad():
            for i, (inp, lbl) in enumerate(tqdm(val_loader)):
                inp, lbl = (Variable(inp.cuda()),
                                Variable(lbl.cuda()))
                out = model(inp)

                inp, out, lbl = to_numpy(inp, out, lbl)
                update_tensorboard(epoch, tb, i, inp, out, lbl)
                val_psnr += compute_PSNR(out, lbl)

            val_psnr = val_psnr/len(val_set)
            print('Validate PSNR: %.4fdB' %val_psnr)
            if val_psnr > best_val_psnr:
                best_val_psnr = val_psnr
                best_epoch = epoch
                print('Saving new best model')
                model_path = os.path.join(check_point, 'best_model.pt')
                if torch.cuda.device_count() > 1:
                    torch.save(model.module.state_dict(), model_path)
                else:
                    torch.save(model.state_dict(), model_path)
            print('End epoch %d, best val PSNR: %.4fdB at epoch: %d' %(epoch, best_val_psnr, best_epoch))
            print()
            tb.add_scalar('Pretrain val PSNR', val_psnr, epoch)
            tb.add_scalar('Pretrain best val PSNR', best_val_psnr, epoch)
if __name__ == '__main__':
    main()
