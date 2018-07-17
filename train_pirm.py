from __future__ import division
from __future__ import print_function
from data import *
from model import *
from solver import *
from utils import *
import torch
import torch.nn.functional as F
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
import random

parser = argparse.ArgumentParser(description='SR benchmark')

# dataset
parser.add_argument('--scale', type=int, default=4, 
                    help='interpolation scale. Default 4')
parser.add_argument('--train_dataset', type=str, default='DIV2K900',
                    help='Training dataset')
parser.add_argument('--valid_dataset', type=str, default='PIRM',
                    help='Training dataset')
parser.add_argument('--num_valids', type=int, default=10,
                    help='Number of image for validation')
# model
parser.add_argument('--num_channels', type=int, default = 256,
                    help='number of resnet channel')
parser.add_argument('--num_blocks', type=int, default = 32,
                    help='number of resnet blocks')
parser.add_argument('--res_scale', type=float, default=0.1)
parser.add_argument('--load', type=str, default='',
                    help='load pretrained model')

# training
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch size used for training. Default 16')
parser.add_argument('--learning_rate', type=float, default=5e-5,
                    help='learning rate used for training. Default 1e-4')
parser.add_argument('--num_epochs', type=int, default=200,
                    help='number of training epochs. Default 100')
parser.add_argument('--num_repeats', type=int, default=20)
parser.add_argument('--patch_size', type=int, default=24,
                    help='input patch size')

# checkpoint
parser.add_argument('--check_point', type=str, default='check_point')

# Loss
parser.add_argument('--alpha_vgg', type=float, default=50)
parser.add_argument('--alpha_gan', type=float, default=1)
parser.add_argument('--alpha_tv', type=float, default=1e-6)
parser.add_argument('--gan_type', type=str, default='RSGAN')
parser.add_argument('--GP', dest='GP', action='store_true',
                    help='Gradient penalty for training GAN (Note: default False)')
parser.add_argument('--fl_gamma', type=float, default=1,
                    help='Focal loss gamma')

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

def main(argv=None):
    # ============Dataset===============
    print('Loading dataset...')
    train_set_path = os.path.join('data/original_data/train', args.train_dataset)
    val_set_path = os.path.join('data/original_data/valid', args.valid_dataset)

    train_set = SRDataset(train_set_path, patch_size=args.patch_size, num_repeats=args.num_repeats, 
                              scale=args.scale, is_aug=True, crop_type='random')
    val_set = SRDataset(val_set_path, patch_size=None, num_repeats=1, scale=args.scale, is_aug=False, 
                            fixed_length=10)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=1,
                            shuffle=False, num_workers=4, pin_memory=True)
    

    # ============Model================
    n_GPUs = torch.cuda.device_count()
    print('Loading model using %d GPU(s)...' %n_GPUs)

    opt = {'patch_size': args.patch_size, 
           'num_channels': args.num_channels, 
           'depth': args.num_blocks, 
           'res_scale': args.res_scale}
    G = Generator(opt)
    if args.load != '':
        model_path = os.path.join('check_point/pretrain/', '{}/c{}_d{}'.format(args.load, args.num_channels, args.num_blocks), 'best_model.pt')
        print('Loading model', model_path)
        G.load_state_dict(torch.load(model_path))
    G = nn.DataParallel(G).cuda()

    D = nn.DataParallel(Discriminator(opt)).cuda()
    vgg = nn.DataParallel(VGG()).cuda()
    cudnn.benchmark = True
        
    check_point = args.check_point
    #clean_and_mk_dir(check_point)

    #========== Optimizer============
    trainable = filter(lambda x: x.requires_grad, G.parameters())
    optim_G = optim.Adam(trainable, betas=(0.9, 0.999),
                         lr=args.learning_rate)
    optim_D = optim.Adam(D.parameters(), betas=(0.9, 0.999), lr=args.learning_rate)
    scheduler_G = lr_scheduler.StepLR(optim_G, step_size=150, gamma=0.5)
    scheduler_D = lr_scheduler.StepLR(optim_D, step_size=150, gamma=0.5)
    
    # ============Loss==============
    l1_loss_fn = nn.L1Loss()
    bce_loss_fn = nn.BCEWithLogitsLoss()
    f_loss_fn = FocalLoss(args.fl_gamma)
    def vgg_loss_fn(output, label):
        vgg_sr, vgg_hr = vgg(output, label)
        return F.mse_loss(vgg_sr, vgg_hr)
    def tv_loss_fn(y):
        loss_var = torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + \
                   torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))
        return loss_var

    # ==========Logging =======
    tb = SummaryWriter(args.check_point)

    # ==========GAN vars======================
    target_real = Variable(torch.Tensor(args.batch_size, 1).fill_(1.0), requires_grad=False).cuda()
    target_fake = Variable(torch.Tensor(args.batch_size, 1).fill_(0.0), requires_grad=False).cuda()

    # Training and validating
    for epoch in range(args.num_epochs):
        scheduler_G.step()
        scheduler_D.step()
        cur_lr = optim_G.param_groups[0]['lr']
        print('Model {}. Epoch [{}/{}]. Learning rate: {}'.format(
            args.check_point, epoch, args.num_epochs, cur_lr))
        
        num_batches = len(train_set)//args.batch_size
        running_loss = np.zeros(6)

        for i, (inputs, labels) in enumerate(tqdm(train_loader)):
            lr, hr = (Variable(inputs.cuda()),
                      Variable(labels.cuda()))

            #######################################
            # Discriminator
            # hr: real, sr: fake
            #######################################
            for p in D.parameters():
                p.requires_grad = True
            optim_D.zero_grad()
            pred_real = D(hr)

            sr = G(lr)
            pred_fake = D(sr.detach())

            if args.gan_type == 'SGAN':
                total_D_loss = bce_loss_fn(pred_real, target_real) + bce_loss_fn(pred_fake, target_fake)
            elif args.gan_type == 'RSGAN':
                total_D_loss = bce_loss_fn(pred_real - pred_fake, target_real)
            elif args.gan_type == 'RaSGAN':
                total_D_loss = (bce_loss_fn(pred_real - torch.mean(pred_fake), target_real) + \
                                bce_loss_fn(pred_fake - torch.mean(pred_real), target_fake))/2
            elif args.gan_type == 'RaLSGAN':
                total_D_loss = (torch.mean((pred_real - torch.mean(pred_fake) - target_real)**2) + \
                                torch.mean((pred_fake - torch.mean(pred_real) + target_real)**2))/2
	    
            # gradient penalty
            if args.GP:
                grad_outputs = torch.ones(args.batch_size, 1).cuda()
                u = torch.FloatTensor(args.batch_size, 1, 1, 1).cuda()
                u.uniform_(0, 1)
                x_both = (hr*u + sr*(1-u)).cuda()
                x_both = Variable(x_both, requires_grad=True)
                grad = torch.autograd.grad(outputs=D(x_both), inputs=x_both,
                                           grad_outputs=grad_outputs, retain_graph=True,
                                           create_graph=True, only_inputs=True)[0]
                grad_penalty = 10*((grad.norm(2, 1).norm(2, 1).norm(2, 1) - 1) ** 2).mean()
                total_D_loss = total_D_loss + grad_penalty

            total_D_loss.backward()
            optim_D.step()

            ######################################
            # Generator
            ######################################
            for p in D.parameters():
                p.requires_grad = False
            optim_G.zero_grad()
            pred_fake = D(sr)
            pred_real = D(hr)

            vgg_loss = vgg_loss_fn(sr, hr)*args.alpha_vgg
            tv_loss = tv_loss_fn(sr)*args.alpha_tv

            if args.gan_type == 'SGAN':
                G_loss = bce_loss_fn(pred_fake, target_real)
            elif args.gan_type == 'RSGAN':
                G_loss = f_loss_fn(pred_fake - pred_real, target_real) #Focal loss
            elif args.gan_type == 'RaSGAN':
                G_loss = (bce_loss_fn(pred_real - torch.mean(pred_fake), target_fake) + \
                          bce_loss_fn(pred_fake_H - torch.mean(pred_real_H), target_real))/2
            elif args.gan_type == 'RaLSGAN':
                G_loss = (torch.mean((pred_real - torch.mean(pred_fake) + target_real)**2) + \
                          torch.mean((pred_fake - torch.mean(pred_real) - target_real)**2))/2
            G_loss = G_loss*args.alpha_gan

            total_G_loss = vgg_loss + G_loss + tv_loss

            total_G_loss.backward()
            optim_G.step()

            # update log
            running_loss += [vgg_loss.item(), 
                             G_loss.item(), 
                             tv_loss.item(),
                             total_D_loss.item(),
                             F.sigmoid(pred_real).mean().item(), 
                             F.sigmoid(pred_fake).mean().item()]

        avr_loss = running_loss/num_batches
        tb.add_scalar('Learning rate', cur_lr, epoch)
        tb.add_scalar('VGG Loss', avr_loss[0], epoch)
        tb.add_scalar('G Loss', avr_loss[1], epoch)
        tb.add_scalar('TV Loss', avr_loss[2], epoch)
        tb.add_scalar('D Loss', avr_loss[3], epoch)
        tb.add_scalar('Real D output', avr_loss[4], epoch)
        tb.add_scalar('Fake D output', avr_loss[5], epoch)
        tb.add_scalar('Total Loss', avr_loss[0:3].sum(), epoch)

        #==========Validate======================
        print('Validating...')
        val_psnr = 0
        num_batches = len(val_set)

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(tqdm(val_loader)):
                lr, hr = (Variable(inputs.cuda()),
                          Variable(labels.cuda()))

                sr = G(lr)
                lr, sr, hr = to_numpy(lr, sr, hr)
                update_tensorboard(epoch, tb, i, lr, sr, hr)
                val_psnr += compute_PSNR(hr, sr)

        val_psnr = val_psnr/num_batches
        tb.add_scalar('Validate PSNR', val_psnr, epoch)
        print('Saving model')
        model_path = os.path.join(check_point, str(epoch+1)+'.pt')
        if n_GPUs > 1:
            torch.save(G.module.state_dict(), model_path)
        else:
            torch.save(G.state_dict(), model_path)
if __name__ == '__main__':
    main()
