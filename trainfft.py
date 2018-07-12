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
parser.add_argument('-s', '--scale', metavar='S', type=int, default=3, 
                    help='interpolation scale. Default 3')
parser.add_argument('-b', '--batch-size', metavar='B', type=int, default=16,
                    help='batch size used for training. Default 16')
parser.add_argument('-l', '--learning-rate', metavar='L', type=float, default=1e-4,
                    help='learning rate used for training. Default 1e-4')
parser.add_argument('-n', '--num-epochs', metavar='N', type=int, default=100,
                    help='number of training epochs. Default 100')
parser.add_argument('--num-repeats', metavar='V', type=int, default=20)
parser.add_argument('--patch-size', metavar='P', type=int, default=48,
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
parser.add_argument('-d', '--num-blocks', metavar='N', type=int, default = 16)


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
                              scale=args.scale, is_aug=True, crop_type='fixed')
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

    G_L2H = Generator_L2H(opt).cuda()
    G_L2H.load_state_dict(torch.load(args.check_point + '/EDSR_baseline_x4.pt'))
    D_H = nn.DataParallel(Discriminator_H(opt)).cuda()
    VGG = nn.DataParallel(PretrainedVGG('54', 255)).cuda()
    fft = FFT().cuda()
        
    check_point = args.check_point
    #clean_and_mk_dir(check_point)

    #========== Optimizer============
    trainable = filter(lambda x: x.requires_grad, G_L2H.parameters())
    optim_G_L2H = optim.Adam(trainable,
                           lr=args.learning_rate)
    optim_D_H = optim.Adam(D_H.parameters(), lr=args.learning_rate)
    scheduler_G_L2H = lr_scheduler.StepLR(optim_G_L2H, step_size=50, gamma=0.5)
    scheduler_D_H = lr_scheduler.StepLR(optim_D_H, step_size=50, gamma=0.5)
    
    # ============Loss==============
    l1_loss_fn = nn.L1Loss()
    bce_loss_fn = nn.BCEWithLogitsLoss()
    def vgg_loss_fn(output, label):
        vgg_sr, vgg_hr = VGG(output, label)
        return F.mse_loss(vgg_sr, vgg_hr)
    def fft_loss_fn(output, label):
        fft_sr, fft_hr = fft(output, label)
        return F.mse_loss(fft_sr, fft_hr)

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
        cur_lr =optim_G_L2H.param_groups[0]['lr']
        print('Model {}. Epoch {}. Learning rate: {}'.format(
            args.check_point, epoch, cur_lr))
        
        num_batches = len(train_set)//args.batch_size

        running_loss = 0

        for i, (inputs, labels) in enumerate(tqdm(train_loader)):
            real_L, real_H = (Variable(inputs.cuda()),
                              Variable(labels.cuda()))

            # Discriminator
            #optim_D_H.zero_grad()
            #pred_real_H = D_H(real_H)
            #D_real_loss = bce_loss_fn(pred_real_H, target_real)

            #fake_H = G_L2H(real_L)
            #pred_fake_H = D_H(fake_H.detach())
            #D_fake_loss = bce_loss_fn(pred_fake_H, target_fake)
            #total_D_loss = D_real_loss + D_fake_loss
            #
            #total_D_loss.backward()
            #optim_D_H.step()

            # Generator
            optim_G_L2H.zero_grad()
            fake_H = G_L2H(real_L)
            #pred_fake_H = D_H(fake_H)

            #vgg_loss = vgg_loss_fn(fake_H, real_H)
            #G_loss = bce_loss_fn(pred_fake_H, target_real)
            #total_G_loss = 5*vgg_loss + 0.15*G_loss
            loss = fft_loss_fn(fake_H, real_H)

            loss.backward()
            optim_G_L2H.step()

            # update log
            running_loss += loss.item()
            print('check loss', loss.item())
            #tb.add_scalar('Train VGG Loss', vgg_loss.item(), epoch*num_batches+i)
            #if i == 10: break

        avr_loss = running_loss/num_batches
        print('fasdfkj', avr_loss)
        #tb.add_scalar('Learning rate', cur_lr, epoch)
        #tb.add_scalar('Train VGG Loss', avr_loss[0], epoch)
        #tb.add_scalar('Train G Loss', avr_loss[1], epoch)
        #tb.add_scalar('Train D Loss', avr_loss[2], epoch)
        tb.add_scalar('Train Total Loss', avr_loss, epoch)
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
            for i, (inputs, labels) in enumerate(tqdm(val_loader)):
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
