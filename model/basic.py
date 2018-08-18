import torch.nn as nn
import torch

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
        bn=True, act=nn.ReLU(True), sn=True):

        #conv = Conv(in_channels, out_channels, kernel_size, stride, bias)
        if sn: conv = spectral_norm(Conv(in_channels, out_channels, kernel_size, stride, bias))
        else: conv = Conv(in_channels, out_channels, kernel_size, stride, bias)
        m = [conv]

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

