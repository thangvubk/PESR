import torch.nn as nn

class PixelDeshuffle(nn.Module):
    def __init__(self, upscale_factor):
        super(PixelDeshuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        return self.pixel_deshuffle(x, self.upscale_factor)

    def __repr__(self):
        return self.__class__.__name__ + '(upscale_factor=' + str(self.upscale_factor) + ')'

    def pixel_deshuffle(self, x, S):
        batch_size, C, HS, WS = x.size()
        CS = C*(S ** 2)

        H = HS//S
        W = WS//S

        x_view = x.contiguous().view(
                    batch_size, C, H, S, W, S)

        shuffle_out = x_view.permute(0, 1, 3, 5, 2, 4).contiguous()
        return shuffle_out.view(batch_size, CS, H, W)
