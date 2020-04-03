import torch.nn.functional as F
from .unet_parts import *
from .aspp import *


class SpoonNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(SpoonNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc1 = DoubleConv(n_channels, 64, 1)
        # self.inc1 = DoubleConv(32, 64, 1)
        self.inc2 = DoubleConv(64, 128, 1)
        # self.inc3 = DoubleConv(128, 128, 1)
        self.inc3 = DoubleConv(128, 64, 1)
        self.inc4 = DoubleConv(64, 3, 1)
        self.down1 = Down(3, 3*32, 3, 3)
        self.down2 = Down(3*32, 3*64, 3, 3)
        self.up1 = Up(9*32, 3*32, 3, 3)
        self.up2 = Up(3*32+3, 3, 1)
        # self.Att = Attention_block_groups(F_g=3*32,F_l=3)
        self.outc1 = OutConv(3, n_classes)
        self.outc2 = OutConv(3, n_classes)

    def forward(self, x):
        x1 = self.inc4(self.inc3(self.inc2(self.inc1(x))))
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        # x = self.Att(x, x1)
        logits = self.outc1(x)
        return [logits, x1]
        # return logits
