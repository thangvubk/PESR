import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.nn.init as init
import torchvision.models as models
from model.pixel_deshuffle import *
#import pytorch_fft.fft.autograd as fft
from torch.autograd import Variable

def default_conv(in_channelss, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channelss, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size//2), stride=stride, bias=bias)
        ]
        if bn: m.append(nn.BatchNorm2d(out_channels))
        if act is not None: m.append(act)
        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)

        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        modules = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                modules.append(conv(n_feat, 4 * n_feat, 3, bias))
                modules.append(nn.PixelShuffle(2))
                if bn: modules.append(nn.BatchNorm2d(n_feat))
                if act: modules.append(act())
        elif scale == 3:
            modules.append(conv(n_feat, 9 * n_feat, 3, bias))
            modules.append(nn.PixelShuffle(3))
            if bn: modules.append(nn.BatchNorm2d(n_feat))
            if act: modules.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*modules)

class Downsampler(nn.Sequential):
    def __init__(self, n_feat, conv=default_conv):
        modules = [
            conv(n_feat, n_feat//4, 3),
            PixelDeshuffle(2),
            conv(n_feat, n_feat//4, 3),
            PixelDeshuffle(2)
        ]
        super(Downsampler, self).__init__(*modules)

class PretrainedVGG1(nn.Module):
    def __init__(self, conv_index, rgb_range=1):
        super(PretrainedVGG1, self).__init__()
        vgg_features = models.vgg19(pretrained=True).features
        modules = [m for m in vgg_features]
        if conv_index == '22':
            self.vgg = nn.Sequential(*modules[:8])
        elif conv_index == '54':
            self.vgg = nn.Sequential(*modules[:35])

        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range)
        self.sub_mean = MeanShift(rgb_range, vgg_mean, vgg_std)
        self.vgg.requires_grad = False

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.vgg(x)
        return x

class PretrainedVGG(nn.Module):
    def __init__(self, conv_index, rgb_range=1):
        super(PretrainedVGG, self).__init__()
        vgg_features = models.vgg19(pretrained=True).features
        modules = [m for m in vgg_features]
        if conv_index == '22':
            self.vgg = nn.Sequential(*modules[:8])
        elif conv_index == '54':
            self.vgg = nn.Sequential(*modules[:35])

        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range)
        self.sub_mean = MeanShift(rgb_range, vgg_mean, vgg_std)
        self.vgg.requires_grad = False

    def forward(self, sr, hr):
        def _forward(x):
            x = self.sub_mean(x)
            x = self.vgg(x)
            return x
            
        vgg_sr = _forward(sr)
        with torch.no_grad():
            vgg_hr = _forward(hr.detach())

        #loss = F.mse_loss(vgg_sr, vgg_hr)

        return vgg_sr, vgg_hr

class Generator_L2H(nn.Module):
    def __init__(self, opt, conv=default_conv):
        super(Generator_L2H, self).__init__()

        n_resblock = opt['depth']
        n_feats = opt['num_channels']
        res_scale = opt['res_scale']
        kernel_size = 3
        scale = opt['scale']
        #act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        act = nn.ReLU(True)
        #act = nn.PReLU(n_feats)

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        rgb_range = 255
        self.sub_mean = MeanShift(rgb_range, rgb_mean, rgb_std)

        # define head module
        modules_head = [conv(3, n_feats, kernel_size)]

        # define body module
        modules_body = [
                        ResBlock(
                        conv, n_feats, kernel_size, act=act, res_scale=res_scale) \
                        for _ in range(n_resblock)]
        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
                        Upsampler(conv, scale, n_feats, act=False),
                        conv(n_feats, 3, kernel_size)]

        self.add_mean = MeanShift(rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        #print(x[0, 0, :, :])
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                            'whose dimensions in the model are {} and '
                                            'whose dimensions in the checkpoint are {}.'
                                            .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'.format(name))


class Generator_H2L(nn.Module):
    def __init__(self, opt, conv=default_conv):
        super(Generator_H2L, self).__init__()

        n_resblock = opt['depth']
        n_feats = opt['num_channels']
        res_scale = opt['res_scale']
        kernel_size = 3
        scale = opt['scale']
        act = nn.ReLU(True)

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        rgb_range = 255
        self.sub_mean = MeanShift(rgb_range, rgb_mean, rgb_std) 

        # define head module
        modules_head = [conv(3, n_feats, kernel_size),
                        Downsampler(n_feats)]

        # define body module
        modules_body = [
                        ResBlock(
                        conv, n_feats, kernel_size, act=act, res_scale=res_scale) \
                        for _ in range(n_resblock)]
        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
                        conv(n_feats, 3, kernel_size)]

        self.add_mean = MeanShift(rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x



class Discriminator_H(nn.Module):
    def __init__(self, opt, gan_type='GAN'):
        super(Discriminator_H, self).__init__()

        in_channels = 3
        out_channels = 64
        depth = 7
        #bn = not gan_type == 'WGAN_GP'
        bn = True
        act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        n_colors = 3
        patch_size = 96

        m_features = [
            BasicBlock(n_colors, out_channels, 3, bn=bn, act=act)
        ]
        for i in range(depth):
            in_channels = out_channels
            if i % 2 == 1:
                stride = 1
                out_channels *= 2
            else:
                stride = 2
            m_features.append(BasicBlock(
                in_channels, out_channels, 3, stride=stride, bn=bn, act=act
            ))

        self.features = nn.Sequential(*m_features)

        patch_size = patch_size // (2**((depth + 1) // 2))
        m_classifier = [
            nn.Linear(out_channels * patch_size**2, 1024),
            act,
            nn.Linear(1024, 1)
        ]
        self.classifier = nn.Sequential(*m_classifier)

    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features.view(features.size(0), -1))

        return output

class Discriminator_L(nn.Module):
    def __init__(self, opt, gan_type='GAN'):
        super(Discriminator_L, self).__init__()

        in_channels = 3
        out_channels = 64
        depth = 7
        #bn = not gan_type == 'WGAN_GP'
        bn = True
        act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        n_colors = 3
        patch_size = 24

        m_features = [
            BasicBlock(n_colors, out_channels, 3, bn=bn, act=act)
        ]
        for i in range(depth):
            in_channels = out_channels
            if i % 4 != 1:
                stride = 1
                out_channels *= 2
            else:
                stride = 2
            m_features.append(BasicBlock(
                in_channels, out_channels, 3, stride=stride, bn=bn, act=act
            ))

        self.features = nn.Sequential(*m_features)

        #patch_size = patch_size // (2**((depth + 1) // 2))
        patch_size = 6
        m_classifier = [
            nn.Linear(out_channels * patch_size**2, 1024),
            act,
            nn.Linear(1024, 1)
        ]
        self.classifier = nn.Sequential(*m_classifier)

    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features.view(features.size(0), -1))

        return output

#class FFT(nn.Module):
#    def __init__(self):
#        super(FFT, self).__init__()
#        self.fft = fft.Fft2d()
#
#    def forward(self, sr, hr):
#        def _forward(img):
#            #fft_re = Variable(torch.empty_like(img))
#            #fft_im = Variable(torch.empty_like(img))
#            #img_im = Variable(torch.zeros_like(img))
#            #for i in range(3):
#            #    fft_re[:, i, :, :], fft_im[:, i, :, :] = \
#            #        self.fft(img[:, i, :, :], img_im[:, i, :, :])
#            #return torch.cat([fft_re, fft_im], 1)
#            spectrum = Variable(torch.empty_like(img))
#            img_im = Variable(torch.zeros_like(img))
#            for i in range(3):
#                fft_re, fft_im = self.fft(img[:, i, :, :], img_im[:, i, :, :])
#                spectrum[:, i, :, :] = torch.log(torch.sqrt(fft_re**2 + fft_im**2))
#            return spectrum
#
#
#        fft_sr = _forward(sr)
#        fft_hr = _forward(hr)
#        return fft_sr, fft_hr

        









