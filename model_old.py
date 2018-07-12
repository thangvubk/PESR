import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.nn.init as init
from pixel_deshuffle import PixelDeshuffle

def get_model(model, opt=None):
    if model == 'EDSR_org':
        return EDSR_org(opt)
    else:
        raise Exception('Unknown model %s' %model)

# original EDSR
def default_conv(in_channelss, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channelss, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, sign):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.bias.data = float(sign) * torch.Tensor(rgb_mean) * rgb_range

        # Freeze the MeanShift layer
        for params in self.parameters():
            params.requires_grad = False

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

class EDSR_org1(nn.Module):
    def __init__(self, opt, conv=default_conv):
        super(EDSR_org, self).__init__()

        n_resblock = opt['depth']
        n_feats = opt['num_channels']
        res_scale = opt['res_scale']
        kernel_size = 3 
        scale = opt['scale']
        act = nn.ReLU(True)

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_range = 255
        self.sub_mean = MeanShift(rgb_range, rgb_mean, -1)
        
        # define head module
        modules_head = [conv(3, n_feats, kernel_size)]

        # define body module
        modules_body1 = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale) \
            for _ in range(n_resblock//2)]

        
        modules_body2 = [
            ResBlock(
                conv, 4*n_feats, kernel_size, act=act, res_scale=res_scale) \
            for _ in range(n_resblock//2)]

        modules_body3 = [ 
            ResBlock(
                conv, 4*n_feats, 1, act=act, res_scale=res_scale) \
            for _ in range(n_resblock)]

        # define tail module
        modules_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, 3, kernel_size)]

        self.add_mean = MeanShift(rgb_range, rgb_mean, 1)
        
        self.head = nn.Sequential(*modules_head)
        self.body1 = nn.Sequential(*modules_body1)
        self.body2 = nn.Sequential(*modules_body2)
        self.body3 = nn.Sequential(*modules_body3)
        self.tail = nn.Sequential(*modules_tail)
        self.downsample = PixelDeshuffle(2)
        self.upsample = nn.PixelShuffle(2)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body1(x)
        res = self.downsample(res)
        res = self.body2(res)
        res = self.body3(res)
        res = self.upsample(res)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x 



class EDSR_org(nn.Module):
    def __init__(self, opt, conv=default_conv):
        super(EDSR_org, self).__init__()

        n_resblock = opt['depth']
        n_feats = opt['num_channels']
        res_scale = opt['res_scale']
        kernel_size = 3
        scale = opt['scale']
        act = nn.ReLU(True)

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_range = 255
        self.sub_mean = MeanShift(rgb_range, rgb_mean, -1)

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

        self.add_mean = MeanShift(rgb_range, rgb_mean, 1)

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
