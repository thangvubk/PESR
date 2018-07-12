from __future__ import division
from __future__ import print_function
from data import *
from model import *
from solver import *
from utils import *
import progressbar
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

parser = argparse.ArgumentParser(description='SR benchmark')
parser.add_argument('-m', '--model', metavar='M', type=str, default='VDSR',
                    help='network architecture. Default SRCNN')
parser.add_argument('-s', '--scale', metavar='S', type=int, default=4, 
                    help='interpolation scale. Default 4')
parser.add_argument('-b', '--batch-size', metavar='B', type=int, default=16,
                    help='batch size used for training. Default 16')
parser.add_argument('-l', '--learning-rate', metavar='L', type=float, default=5e-5,
                    help='learning rate used for training. Default 1e-4')
parser.add_argument('-n', '--num-epochs', metavar='N', type=int, default=50,
                    help='number of training epochs. Default 100')
parser.add_argument('--num-repeats', metavar='V', type=int, default=20)
parser.add_argument('--patch-size', metavar='P', type=int, default=24,
                    help='input patch size')
parser.add_argument('--num-valids', metavar='N', type=int, default=10,
                    help='Number of image for validation')
parser.add_argument('-p', '--precision', metavar='P', type=str, default='single',
                    help='precision using for training')
parser.add_argument('-f', '--finetune', dest='finetune', action='store_true',
                    help='fine tune the model under check_point dir,\
                    instead of training from scratch. Default False')
parser.add_argument('-t', '--dataset', metavar='T', type=str, default='DIV2K',
                    help='Training dataset')
parser.add_argument('-r', '--res-scale', metavar='R', type=float, default=1)
parser.add_argument('--check-point', type=str, default='check_point')
#temp ssim tuning
parser.add_argument('-a', '--alpha', metavar='A', type=str, default="") 
parser.add_argument('-c', '--num-channels', metavar='N', type=int, default = 64)
parser.add_argument('-d', '--num_blocks', metavar='N', type=int, default = 16)
parser.add_argument('--alpha_vgg', type=float, default=5)
parser.add_argument('--alpha_gan', type=float, default=0.1)
parser.add_argument('--alpha_cyc', type=float, default=0.15)
parser.add_argument('--gan-type', type=str, default='SGAN')

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
    train_set_path = os.path.join('data/original_data/train', args.dataset)
    val_set_path = os.path.join('data/original_data/valid', args.dataset)

    train_set = DIV2K_Dataset(train_set_path, patch_size=args.patch_size, num_repeats=args.num_repeats, 
                              scale=args.scale, is_aug=True, crop_type='random')
    val_set = DIV2K_Dataset(val_set_path, patch_size=200, num_repeats=1, scale=args.scale, is_aug=False, 
                            fixed_length=10)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=1,
                            shuffle=False, num_workers=4, pin_memory=True)
    

    # ============Model================
    n_GPUs = torch.cuda.device_count()
    print('Loading model using %d GPU(s)...' %n_GPUs)
    opt = {'scale': args.scale, 'num_channels': args.num_channels, 'depth': args.num_blocks, 'res_scale': args.res_scale}
    G_L2H = Generator_L2H(opt)
    G_L2H.load_state_dict(torch.load('check_point/pretrain_LeakyRL/c64_d16_/best_model.pt'))
    G_L2H = nn.DataParallel(G_L2H).cuda()
    D_H = nn.DataParallel(Discriminator_H(opt)).cuda()
    VGG = nn.DataParallel(PretrainedVGG('54', 255)).cuda()
        
    check_point = args.check_point
    #clean_and_mk_dir(check_point)

    #========== Optimizer============
    trainable = filter(lambda x: x.requires_grad, G_L2H.parameters())
    optim_G_L2H = optim.Adam(trainable,
                           lr=args.learning_rate)
    optim_D_H = optim.Adam(D_H.parameters(), lr=args.learning_rate)
    scheduler_G_L2H = lr_scheduler.StepLR(optim_G_L2H, step_size=30, gamma=0.5)
    scheduler_D_H = lr_scheduler.StepLR(optim_D_H, step_size=30, gamma=0.5)
    
    # ============Loss==============
    l1_loss_fn = nn.L1Loss()
    bce_loss_fn = nn.BCEWithLogitsLoss()
    def vgg_loss_fn(output, label):
        vgg_sr, vgg_hr = VGG(output, labels)
        return F.mse_loss(vgg_sr, vgg_hr)

    # ==========Log and book-keeping vars =======
    tb = SummaryWriter(args.check_point)
    (best_val_error, best_epoch) = (1e6, -1)

    # ==========GAN vars======================
    target_real = Variable(torch.Tensor(args.batch_size, 1).fill_(1.0), requires_grad=False).cuda()
    target_fake = Variable(torch.Tensor(args.batch_size, 1).fill_(0.0), requires_grad=False).cuda()

    # Training and validating
    for epoch in range(args.num_epochs):
        scheduler_G_L2H.step()
        scheduler_D_H.step()
        cur_lr = optim_G_L2H.param_groups[0]['lr']
        print('Model {}. Epoch {}. Learning rate: {}'.format(
            args.check_point, epoch, cur_lr))
        
        num_batches = len(train_set)//args.batch_size

        running_loss = np.zeros(5)

        for i, (inputs, labels) in enumerate(tqdm(train_loader)):
            real_L, real_H = (Variable(inputs.cuda()),
                              Variable(labels.cuda()))

            # Discriminator
            for p in D_H.parameters():
                p.requires_grad = True
            optim_D_H.zero_grad()
            pred_real_H = D_H(real_H)

            fake_H = G_L2H(real_L)
            pred_fake_H = D_H(fake_H.detach())

            if args.gan_type == 'SGAN':
                total_D_loss = bce_loss_fn(pred_real_H, target_real) + bce_loss_fn(pred_fake_H, target_fake)
            elif args.gan_type == 'RSGAN':
                total_D_loss = bce_loss_fn(pred_real_H - pred_fake_H, target_real)
            elif args.gan_type == 'RaSGAN':
                total_D_loss = (bce_loss_fn(pred_real_H - torch.mean(pred_fake_H), target_real) + bce_loss_fn(pred_fake_H - torch.mean(pred_real_H), target_fake))/2
            elif args.gan_type == 'RaLSGAN':
                total_D_loss = (torch.mean((pred_real_H - torch.mean(pred_fake_H) - target_real)**2) + \
                        torch.mean((pred_fake_H - torch.mean(pred_real_H) + target_real)**2))/2
            total_D_loss.backward()
            optim_D_H.step()

            # Generator
            for p in D_H.parameters():
                p.requires_grad = False
            optim_G_L2H.zero_grad()
            pred_fake_H = D_H(fake_H)
            pred_real_H = D_H(real_H)

            vgg_loss = vgg_loss_fn(fake_H, real_H)*args.alpha_vgg
            if args.gan_type == 'SGAN':
                G_loss = bce_loss_fn(pred_fake_H, target_real)
            elif args.gan_type == 'RSGAN':
                G_loss = bce_loss_fn(pred_fake_H - pred_real_H, target_real)
            elif args.gan_type == 'RaSGAN':
                G_loss = (bce_loss_fn(pred_real_H - torch.mean(pred_fake_H), target_fake) + bce_loss_fn(pred_fake_H - torch.mean(pred_real_H), target_real))/2
            elif args.gan_type == 'RaLSGAN':
                G_loss = (torch.mean((pred_real_H - torch.mean(pred_fake_H) + target_real)**2) + \
                        torch.mean((pred_fake_H - torch.mean(pred_real_H) - target_real)**2))/2
            G_loss = G_loss*args.alpha_gan

            total_G_loss = vgg_loss + G_loss

            total_G_loss.backward()
            optim_G_L2H.step()

            # update log
            running_loss += [vgg_loss.item(), G_loss.item(), total_D_loss.item(),
                    F.sigmoid(pred_real_H).mean().item(), F.sigmoid(pred_fake_H).mean().item()]
            #tb.add_scalar('Train VGG Loss', vgg_loss.item(), epoch*num_batches+i)
            #if i == 10: break

        avr_loss = running_loss/num_batches
        tb.add_scalar('Learning rate', cur_lr, epoch)
        tb.add_scalar('Train VGG Loss', avr_loss[0], epoch)
        tb.add_scalar('Train G Loss', avr_loss[1], epoch)
        tb.add_scalar('Train D Loss', avr_loss[2], epoch)
        tb.add_scalar('Train real D output', avr_loss[3], epoch)
        tb.add_scalar('Train fake D output', avr_loss[4], epoch)
        tb.add_scalar('Train Total Loss', avr_loss[0:2].sum(), epoch)


        print('Validating...')
        val_psnr = 0
        num_batches = len(val_set)

        running_loss = np.zeros(3)

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(tqdm(val_loader)):
                #inputs = Variable(inputs.cuda(), volatile=True)
                real_L, real_H = (Variable(inputs.cuda()),
                                  Variable(labels.cuda()))

                
                # Generator
                fake_H = G_L2H(real_L)
                val_psnr += update_log(epoch, tb, i, real_L, fake_H, real_H)

        val_psnr = val_psnr/num_batches
        print('Validate PSNR: %.4fdB' %val_psnr)
        #if val_psnr > best_val_psnr:
        if True:
            #best_val_psnr = val_psnr
            #best_epoch = epoch
            print('Saving model')
            model_path = os.path.join(check_point, str(epoch)+'.pt')
            if torch.cuda.device_count() > 1:
                torch.save(G_L2H.module.state_dict(), model_path)
            else:
                torch.save(G_L2H.state_dict(), model_path)
        #print('End epoch %d, best val PSNR: %.4fdB at epoch: %d' %(epoch, best_val_psnr, best_epoch))
        #print()
    #log_file.close()       
if __name__ == '__main__':
    main()
