import torch.nn.functional as F
from .unet_parts import *
from .aspp import *


class rendNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(rendNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.groups = 32

        self.inc1 = DoubleConv(n_channels, 64, 1)
        # self.inc1 = DoubleConv(32, 64, 1)
        self.inc2 = DoubleConv(64, 128, 1)
        # self.inc3 = DoubleConv(128, 128, 1)
        self.inc3 = DoubleConv(128, 64, 1)
        self.inc4 = DoubleConv(64, self.groups, 1)
        self.down1 = Down(self.groups, 2*self.groups, 3, self.groups)
        self.down2 = Down(2*self.groups, 4*self.groups, 3, self.groups)
        self.down3 = Down(4*self.groups, 8*self.groups, 3, self.groups)
        self.up = nn.Upsample(scale_factor=8, mode='nearest')
        self.outc = OutConv(9*self.groups, n_classes)

    def forward(self, x):
        x1 = self.inc4(self.inc3(self.inc2(self.inc1(x))))
        x2 = self.up(self.down3(self.down2(self.down1(x1))))
        logits = self.outc(torch.cat([x1, x2], dim=1))
        return logits
