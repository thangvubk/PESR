from __future__ import division
from __future__ import print_function
from data import *
from model import *
from utils import *
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import argparse
import os 
import time
from tqdm import tqdm
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import random
import pdb

parser = argparse.ArgumentParser(description='PIRM 2018')

# dataset
parser.add_argument('--scale', type=int, default=4, 
                    help='interpolation scale. Default 4')
parser.add_argument('--train_dataset', type=str, default='DIV2K',
                    help='Training dataset')
parser.add_argument('--valid_dataset', type=str, default='PIRM',
                    help='Training dataset')
parser.add_argument('--num_valids', type=int, default=10,
                    help='Number of image for validation')
# model
parser.add_argument('--num_channels', type=int, default=256,
                    help='number of resnet channel')
parser.add_argument('--num_blocks', type=int, default=32,
                    help='number of resnet blocks')
parser.add_argument('--res_scale', type=float, default=0.1)
parser.add_argument('--phase', type=str, default='train',
                    help='phase: pretrain or train')
parser.add_argument('--pretrained_model', type=str, default='',
                    help='pretrained model for train phase (optional)')

# training
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch size used for training')
parser.add_argument('--learning_rate', type=float, default=5e-5,
                    help='learning rate used for training (use 1e-4 for pretrain)')
parser.add_argument('--lr_step', type=int, default=120,
                    help='steps to decay learning rate')
parser.add_argument('--num_epochs', type=int, default=200,
                    help='number of training epochs')
parser.add_argument('--num_repeats', type=int, default=20,
                    help='number of repeat per image for each epoch')
parser.add_argument('--patch_size', type=int, default=24,
                    help='input patch size')

# checkpoint
parser.add_argument('--check_point', type=str, default='check_point/my_model',
                    help='path to save log and model')
parser.add_argument('--snapshot_every', type=int, default=10, 
                    help='snapshot freq, used for train model only')

# GAN
parser.add_argument('--gan_type', type=str, default='RSGAN')
parser.add_argument('--GP', type=lambda x: (str(x).lower() == 'true'), default=False,
                    help='Gradient penalty for training GAN (Note: default False)')
parser.add_argument('--spectral_norm', type=lambda x: (str(x).lower() == 'true'), default=False,
                    help='Discriminator Spectral norm')

parser.add_argument('--focal_loss', type=lambda x: (str(x).lower() == 'true'), default=True)
parser.add_argument('--fl_gamma', type=float, default=1,
                    help='Focal loss gamma')
parser.add_argument('--alpha_vgg', type=float, default=50)
parser.add_argument('--alpha_gan', type=float, default=1)
parser.add_argument('--alpha_tv', type=float, default=1e-6)
parser.add_argument('--alpha_l1', type=float, default=0)
args = parser.parse_args()

print('############################################################')
print('# Image Super Resolution - PIRM2018 - TEAM_AIM             #')
print('# Implemented by Thang Vu, thangvubk@gmail.com             #')
print('############################################################')
print('')
print('_____________YOUR SETTINGS_____________')
for arg in vars(args):
    print("%20s: %s" %(str(arg), str(getattr(args, arg))))
print('')

def main(argv=None):
    # ============Dataset===============
    print('Loading dataset...')
    train_set = SRDataset(args.train_dataset, 'train', patch_size=args.patch_size, 
                          num_repeats=args.num_repeats, is_aug=True, crop_type='random')
    val_set = SRDataset(args.valid_dataset, 'valid', patch_size=None, num_repeats=1, 
                        is_aug=False, fixed_length=10)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=1,
                            shuffle=False, num_workers=4, pin_memory=True)

    # ============Model================
    n_GPUs = torch.cuda.device_count()
    print('Loading model using %d GPU(s)' %n_GPUs)

    opt = {'patch_size': args.patch_size, 
           'num_channels': args.num_channels, 
           'depth': args.num_blocks, 
           'res_scale': args.res_scale,
           'spectral_norm': args.spectral_norm}
    G = Generator(opt)
    if args.pretrained_model != '':
        print('Fetching pretrained model', args.pretrained_model)
        G.load_state_dict(torch.load(args.pretrained_model))
    G = nn.DataParallel(G).cuda()

    D = nn.DataParallel(Discriminator(opt)).cuda()

    vgg = nn.DataParallel(VGG()).cuda()

    cudnn.benchmark = True

    #========== Optimizer============
    trainable = filter(lambda x: x.requires_grad, G.parameters())
    optim_G = optim.Adam(trainable, betas=(0.9, 0.999),
                         lr=args.learning_rate)
    optim_D = optim.Adam(D.parameters(), betas=(0.9, 0.999), lr=args.learning_rate)
    scheduler_G = lr_scheduler.StepLR(optim_G, step_size=args.lr_step, gamma=0.5)
    scheduler_D = lr_scheduler.StepLR(optim_D, step_size=args.lr_step, gamma=0.5)
    
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

    # ==========Logging and book-keeping=======
    check_point = os.path.join(args.check_point, args.phase)
    tb = SummaryWriter(check_point)
    best_psnr = 0

    # ==========GAN vars======================
    target_real = Variable(torch.Tensor(args.batch_size, 1).fill_(1.0), requires_grad=False).cuda()
    target_fake = Variable(torch.Tensor(args.batch_size, 1).fill_(0.0), requires_grad=False).cuda()

    # Training and validating
    for epoch in range(1, args.num_epochs+1):
        
        #===========Pretrain===================
        if args.phase == 'pretrain':
            scheduler_G.step()
            cur_lr = optim_G.param_groups[0]['lr']
            print('Model {}. Epoch [{}/{}]. Learning rate: {}'.format(
                args.check_point, epoch, args.num_epochs, cur_lr))
            
            num_batches = len(train_set)//args.batch_size
            running_loss = 0

            for i, (inputs, labels) in enumerate(tqdm(train_loader)):
                lr, hr = (Variable(inputs.cuda()),
                          Variable(labels.cuda()))
                
                sr = G(lr)
                optim_G.zero_grad()

                loss = l1_loss_fn(sr, hr)
                loss.backward()
                optim_G.step()

                # update log
                running_loss += loss.item()

            avr_loss = running_loss/num_batches
            tb.add_scalar('Learning rate', cur_lr, epoch)
            tb.add_scalar('Pretrain Loss', avr_loss, epoch)
            print('Finish train [%d/%d]. Loss: %.2f' %(epoch, args.num_epochs, avr_loss))

        #===============Train======================
        else:
            scheduler_G.step()
            scheduler_D.step()
            cur_lr = optim_G.param_groups[0]['lr']
            print('Model {}. Epoch [{}/{}]. Learning rate: {}'.format(
                check_point, epoch, args.num_epochs, cur_lr))
            
            num_batches = len(train_set)//args.batch_size
            running_loss = np.zeros(5)

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

                l1_loss = l1_loss_fn(sr, hr)*args.alpha_l1
                vgg_loss = vgg_loss_fn(sr, hr)*args.alpha_vgg
                tv_loss = tv_loss_fn(sr)*args.alpha_tv

                if args.gan_type == 'SGAN':
                    if args.focal_loss:
                        G_loss = f_loss_fn(pred_fake, target_real)
                    else:
                        G_loss = bce_loss_fn(pred_fake, target_real)
                elif args.gan_type == 'RSGAN':
                    if args.focal_loss:
                        G_loss = f_loss_fn(pred_fake - pred_real, target_real) #Focal loss
                    else:
                        G_loss = bce_loss_fn(pred_fake - pred_real, target_real)
                G_loss = G_loss*args.alpha_gan

                total_G_loss = l1_loss + vgg_loss + G_loss + tv_loss

                total_G_loss.backward()
                optim_G.step()

                # update log
                running_loss += [l1_loss.item(),
                                 vgg_loss.item(), 
                                 G_loss.item(), 
                                 tv_loss.item(),
                                 total_D_loss.item()]

            avr_loss = running_loss/num_batches
            tb.add_scalar('Learning rate', cur_lr, epoch)
            tb.add_scalar('L1 Loss', avr_loss[0], epoch)
            tb.add_scalar('VGG Loss', avr_loss[1], epoch)
            tb.add_scalar('G Loss', avr_loss[2], epoch)
            tb.add_scalar('TV Loss', avr_loss[3], epoch)
            tb.add_scalar('D Loss', avr_loss[4], epoch)
            tb.add_scalar('Total G Loss', avr_loss[0:4].sum(), epoch)
            print('Finish train [%d/%d]. L1: %.2f. VGG: %.2f. G: %.2f. TV: %.2f. Total G: %.2f. D: %.2f'\
                  %(epoch, args.num_epochs, avr_loss[0], avr_loss[1], avr_loss[2], 
                    avr_loss[3], avr_loss[0:4].sum(), avr_loss[4]))

        #===============Validate================
        print('Validating...')
        val_psnr = 0
        num_batches = len(val_set)

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(tqdm(val_loader)):
                lr, hr = (Variable(inputs.cuda()),
                          Variable(labels.cuda()))

                sr = G(lr)
                update_tensorboard(epoch, tb, i, lr, sr, hr)
                val_psnr += compute_PSNR(hr, sr)

        val_psnr = val_psnr/num_batches
        tb.add_scalar('Validate PSNR', val_psnr, epoch)

        if args.phase == 'pretrain':
            print('Finish valid [%d/%d]. Best PSNR: %.4fdB. Cur PSNR: %.4fdB' \
                  %(epoch, args.num_epochs, best_psnr, val_psnr))
            if best_psnr < val_psnr:
                best_psnr = val_psnr
                model_path = os.path.join(check_point, 'best_model.pt')
                torch.save(G.module.state_dict(), model_path)
                print('Saved new best model.')
        else:
            print('Finish valid [%d/%d]. PSNR: %.4fdB' %(epoch, args.num_epochs, val_psnr))
            if epoch%args.snapshot_every == 0:
                model_path = os.path.join(check_point, 'model_{}.pt'.format(epoch))
                torch.save(G.module.state_dict(), model_path)
                print('Saved snapshot model.')
        print('')

if __name__ == '__main__':
    main()
