from model.basic import *

class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()

        n_resblock = opt['depth']
        n_feats = opt['num_channels']
        res_scale = opt['res_scale']
        kernel_size = 3
        act = nn.ReLU(True)

        rgb_mean = (0.4488, 0.4371, 0.4040) # DIV2K800
        #rgb_mean = (0.4463, 0.4368, 0.4046) # DIV2K900
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
        sn = opt['spectral_norm']

        m_features = [
            BasicBlock(n_colors, out_channels, 3, bn=True, act=act, sn=sn)
        ]
        for i in range(depth):
            in_channels = out_channels
            if i % 2 == 1:
                stride = 1
                out_channels *= 2
            else:
                stride = 2
            m_features.append(BasicBlock(
                in_channels, out_channels, 3, stride=stride, bn=True, act=act, sn=sn
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
