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
                    help='interpolation scale. Default 3')
parser.add_argument('-b', '--batch-size', metavar='B', type=int, default=16,
                    help='batch size used for training. Default 16')
parser.add_argument('-l', '--learning-rate', metavar='L', type=float, default=5e-5,
                    help='learning rate used for training. Default 1e-4')
parser.add_argument('-n', '--num-epochs', metavar='N', type=int, default=200,
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


# Loss function
parser.add_argument('--alpha_vgg', type=float, default=5)
parser.add_argument('--alpha_gan', type=float, default=0.15)
parser.add_argument('--alpha_cyc', type=float, default=0.15)


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
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1,
                            shuffle=False, num_workers=4, pin_memory=True)
    

    # ============Model================
    n_GPUs = torch.cuda.device_count()
    print('Loading model using %d GPU(s)...' %n_GPUs)
    opt = {'scale': args.scale, 'num_channels': args.num_channels, 'depth': args.num_blocks, 'res_scale': args.res_scale}
    G_L2H = nn.DataParallel(Generator_L2H(opt)).cuda()
    D_H = nn.DataParallel(Discriminator_H(opt)).cuda()
    G_H2L = nn.DataParallel(Generator_H2L(opt)).cuda()
    D_L = nn.DataParallel(Discriminator_L(opt)).cuda()
    VGG54 = nn.DataParallel(PretrainedVGG('54', 255)).cuda()
    VGG22 = nn.DataParallel(PretrainedVGG('22', 255)).cuda() 
        
    check_point = args.check_point
    clean_and_mk_dir(check_point)

    #========== Optimizer============
    trainable = filter(lambda x: x.requires_grad, G_L2H.parameters())
    optim_G_L2H = optim.Adam(trainable,
                           lr=args.learning_rate)
    optim_D_H = optim.Adam(D_H.parameters(), lr=args.learning_rate)
    trainable = filter(lambda x: x.requires_grad, G_H2L.parameters())
    optim_G_H2L = optim.Adam(trainable, lr=args.learning_rate)
    optim_D_L = optim.Adam(D_L.parameters(), lr=args.learning_rate)

    
    scheduler_G_L2H = lr_scheduler.StepLR(optim_G_L2H, step_size=150, gamma=0.5)
    scheduler_D_H = lr_scheduler.StepLR(optim_D_H, step_size=150, gamma=0.5)
    scheduler_G_H2L = lr_scheduler.StepLR(optim_G_H2L, step_size=150, gamma=0.5)
    scheduler_D_L = lr_scheduler.StepLR(optim_D_L, step_size=150, gamma=0.5)
    
    # ============Loss==============
    l1_loss_fn = nn.L1Loss()
    bce_loss_fn = nn.BCEWithLogitsLoss()
    def vgg_loss_fn(output, label, conv_index='54'):
        if conv_index == '22':
            vgg_sr, vgg_hr = VGG22(output, label)
        elif conv_index == '54':
            vgg_sr, vgg_hr = VGG54(output, label)
        return F.mse_loss(vgg_sr, vgg_hr)

    # ==========Log and book-keeping vars =======
    tb = SummaryWriter(args.check_point)
    (best_val_psnr, best_epoch) = (-1, -1)

    # ==========GAN vars======================
    target_real = Variable(torch.Tensor(args.batch_size, 1).fill_(1.0), requires_grad=False).cuda()
    target_fake = Variable(torch.Tensor(args.batch_size, 1).fill_(0.0), requires_grad=False).cuda()

    # Training and validating
    for epoch in range(args.num_epochs):
        scheduler_G_L2H.step()
        scheduler_D_H.step()
        scheduler_G_H2L.step()
        scheduler_D_L.step()
        cur_lr =optim_G_L2H.param_groups[0]['lr']
        print('Model {}. Epoch {}. Learning rate: {}'.format(
            args.check_point, epoch, cur_lr))
        
        num_batches = len(train_set)//args.batch_size

        running_loss = np.zeros(8)

        for i, (inputs, labels) in enumerate(tqdm(train_loader, ncols=50)):
            real_L, real_H = (Variable(inputs.cuda()),
                              Variable(labels.cuda()))

            # Discriminator H
            optim_D_H.zero_grad()
            pred_real_H = D_H(real_H)
            D_H_real_loss = bce_loss_fn(pred_real_H, target_real)

            fake_H = G_L2H(real_L)
            pred_fake_H = D_H(fake_H.detach())
            D_H_fake_loss = bce_loss_fn(pred_fake_H, target_fake)
            total_D_H_loss = D_H_real_loss + D_H_fake_loss
            
            total_D_H_loss.backward()
            optim_D_H.step()

            # Discriminator L
            optim_D_L.zero_grad()
            pred_real_L = D_L(real_L)
            D_L_real_loss = bce_loss_fn(pred_real_L, target_real)

            fake_L = G_H2L(real_H)
            pred_fake_L = D_L(fake_L.detach())
            D_L_fake_loss = bce_loss_fn(pred_fake_L, target_fake)

            total_D_L_loss = D_L_fake_loss + D_L_real_loss
            total_D_L_loss.backward()
            optim_D_L.step()

            # Generator
            optim_G_L2H.zero_grad()
            optim_G_H2L.zero_grad()

            pred_fake_H = D_H(fake_H)
            pred_fake_L = D_L(fake_L)

            vgg_H_loss = vgg_loss_fn(fake_H, real_H)*args.alpha_vgg
            G_H_loss = bce_loss_fn(pred_fake_H, target_real)*args.alpha_gan
            
            vgg_L_loss = vgg_loss_fn(fake_L, real_L)*args.alpha_vgg
            G_L_loss = bce_loss_fn(pred_fake_L, target_real)*args.alpha_gan

            rec_L = G_H2L(fake_H)
            rec_H = G_L2H(fake_L)

            cycle_loss_LHL = l1_loss_fn(rec_L, real_L)*args.alpha_cyc
            cycle_loss_HLH = l1_loss_fn(rec_H, real_H)*args.alpha_cyc

            total_G_loss = vgg_H_loss + vgg_L_loss + G_H_loss + G_L_loss + cycle_loss_LHL + cycle_loss_HLH

            total_G_loss.backward()
            optim_G_L2H.step()
            optim_G_H2L.step()

            # update log
            running_loss += [vgg_H_loss.item(), vgg_L_loss.item(), 
                             G_H_loss.item(), G_L_loss.item(),
                             cycle_loss_LHL.item(), cycle_loss_HLH.item(),
                             total_D_H_loss.item(), total_D_L_loss.item()]
            #tb.add_scalar('Train VGG Loss', vgg_loss.item(), epoch*num_batches+i)
            #if i == 10: break

        avr_loss = running_loss/num_batches
        tb.add_scalar('Learning rate', cur_lr, epoch)
        tb.add_scalar('Train VGG H Loss', avr_loss[0], epoch)
        tb.add_scalar('Train VGG L Loss', avr_loss[1], epoch)
        tb.add_scalar('Train G H  Loss', avr_loss[2], epoch)
        tb.add_scalar('Train G L  Loss', avr_loss[3], epoch)
        tb.add_scalar('Train Cyc LHL Loss', avr_loss[4], epoch)
        tb.add_scalar('Train Cyc HLH Loss', avr_loss[5], epoch)
        tb.add_scalar('Train D H  Loss', avr_loss[6], epoch)
        tb.add_scalar('Train D L  Loss', avr_loss[7], epoch)
        tb.add_scalar('Train Total Loss', avr_loss[0:6].sum(), epoch)
        #print('VGG_Loss%d: loss %f' %(epoch, training_loss))

        #print('Saving model at epoch %d' %epoch)
        #model_path = os.path.join(check_point,  'model_{}.pt'.format(epoch))
        #newest_model_path = os.path.join(check_point, 'newest_model.pt')
        #if torch.cuda.device_count() > 1:
        #    torch.save(model.module.state_dict(), model_path)
        #    torch.save(model.module.state_dict(), newest_model_path)
        #else:
        #    torch.save(model.state_dict(), model_path)
        #    torch.save(model.state_dict(), newest_model_path)

        print('Validating...')
        val_psnr = 0
        num_batches = len(val_set)

        running_loss = np.zeros(3)

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(tqdm(val_loader, ncols=50)):
                #inputs = Variable(inputs.cuda(), volatile=True)
                real_L, real_H = (Variable(inputs.cuda()),
                                  Variable(labels.cuda()))

                
                # Generator
                fake_H = G_L2H(real_L)
                #pred_fake_H = D_H(fake_H)

                #vgg_loss = vgg_loss_fn(fake_H, real_H)
                #G_loss = bce_loss_fn(pred_fake_H, target_real)
                #total_G_loss = 5*vgg_loss + 0.15*G_loss

                # Discriminator
                #pred_real_H = D_H(real_H)
                #D_real_loss = bce_loss_fn(pred_real_H, target_real)

                #pred_fake_H = D_H(fake_H.detach())
                #D_fake_loss = bce_loss_fn(pred_fake_H, target_fake)
                #total_D_loss = D_real_loss + D_fake_loss
                
                # update log
                #running_loss += [vgg_loss.item(), G_loss.item(), total_D_loss.item()]

                val_psnr += update_log(epoch, tb, i, real_L, fake_H, real_H)


                #outputs = model(inputs)
                #outputs = outputs.data.cpu().numpy()
                #labels = labels.numpy()
            
                #out = outputs[0, :, :, :]
                #out = out.transpose(1, 2, 0)
                #out = rgb2y(out)
                #lbl = labels[0, :, :, :]
                #lbl = lbl.transpose(1, 2, 0)
                #lbl = rgb2y(lbl)

                #psnr = compute_PSNR(out, lbl)
                #val_psnr += psnr
                #bar.update(i+1, force=True)
        val_psnr = val_psnr/num_batches
        print('Validate PSNR: %.4fdB' %val_psnr)
        #if val_psnr > best_val_psnr:
        #    best_val_psnr = val_psnr
        #    best_epoch = epoch
        #    print('Saving new best model')
        #    model_path = os.path.join(check_point, 'best_model.pt')
        #    if torch.cuda.device_count() > 1:
        #        torch.save(model.module.state_dict(), model_path)
        #    else:
        #        torch.save(model.state_dict(), model_path)
        #print('End epoch %d, best val PSNR: %.4fdB at epoch: %d' %(epoch, best_val_psnr, best_epoch))
        #print()
    log_file.close()       
if __name__ == '__main__':
    main()
