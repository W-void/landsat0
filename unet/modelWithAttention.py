import torch.nn.functional as F
from .unet_parts import *


class UNetWithAttention(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNetWithAttention, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc1 = DoubleConv(n_channels, 32, 1)
        self.inc2 = DoubleConv(32, 64, 1)
        self.inc3 = DoubleConv(64, 64, 1)
        self.inc3 = DoubleConv(64, 32, 1)
        self.inc4 = DoubleConv(32, 16, 1)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 64)
        self.up1 = Up(128, 32, bilinear)
        self.up2 = Up(64, 16, bilinear)
        self.outc = OutConv(16, n_classes)

    def forward(self, x):
        x1 = self.inc1(x)
        x_ = self.inc4(self.inc3(self.inc2(x1)))
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        logits = self.outc(x * x_)
        return logits
