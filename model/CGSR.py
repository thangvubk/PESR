import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.nn.init as init
import torchvision.models as models
from torch.autograd import Variable

class Conv(nn.Conv2d):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, bias=True):
        super(Conv, self).__init__(in_planes, out_planes, kernel_size, 
                padding=(kernel_size//2), stride=stride, bias=bias)

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
        self, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(Conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feats))
            if i == 0: modules_body.append(act)

        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, n_feats):
        super(Upsampler, self).__init__(Conv(n_feats, 4*n_feats, 3),
                                        nn.PixelShuffle(2),
                                        Conv(n_feats, 4*n_feats, 3),
                                        nn.PixelShuffle(2),
                                        Conv(n_feats, 3, 3))


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        vgg_features = models.vgg19(pretrained=True).features
        modules = [m for m in vgg_features]
        self.vgg = nn.Sequential(*modules[:35]) #VGG 5_4

        rgb_range = 255
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

        return vgg_sr, vgg_hr

class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()

        n_resblock = opt['depth']
        n_feats = opt['num_channels']
        res_scale = opt['res_scale']
        kernel_size = 3
        #act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        act = nn.ReLU(True)
        #act = nn.PReLU(n_feats)

        #rgb_mean = (0.4488, 0.4371, 0.4040) # DIV2K800
        rgb_mean = (0.4463, 0.4368, 0.4046) # DIV2K900
        rgb_std = (1.0, 1.0, 1.0)
        rgb_range = 255

        modules_body = [ ResBlock(n_feats, kernel_size, act=act, res_scale=res_scale) \
                          for _ in range(n_resblock)]
        modules_body.append(Conv(n_feats, n_feats, kernel_size))

        self.sub_mean = MeanShift(rgb_range, rgb_mean, rgb_std)
        self.embed = Conv(3, n_feats, kernel_size)
        self.body = nn.Sequential(*modules_body)
        self.upsample = Upsampler(n_feats)
        self.add_mean = MeanShift(rgb_range, rgb_mean, rgb_std, 1)

    def forward(self, x):
        x = self.sub_mean(x)

        x = self.embed(x)
        res = self.body(x)
        res += x

        x = self.upsample(res)
        x = self.add_mean(x)

        return x

class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()

        in_channels = 3
        out_channels = 64
        depth = 7
        act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        n_colors = 3
        patch_size = opt['patch_size']*4

        m_features = [
            BasicBlock(n_colors, out_channels, 3, bn=True, act=act)
        ]
        for i in range(depth):
            in_channels = out_channels
            if i % 2 == 1:
                stride = 1
                out_channels *= 2
            else:
                stride = 2
            m_features.append(BasicBlock(
                in_channels, out_channels, 3, stride=stride, bn=True, act=act
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

