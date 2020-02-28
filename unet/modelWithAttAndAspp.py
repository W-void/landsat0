from .unet_parts import *
from .aspp import ASPP


class UNetWithAttention(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNetWithAttention, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc1 = DoubleConv(n_channels, 64, 1)
        # self.inc1 = DoubleConv(32, 64, 1)
        self.inc2 = DoubleConv(64, 128, 1)
        self.inc3 = DoubleConv(128, 128, 1)
        self.inc3 = DoubleConv(128, 64, 1)
        self.inc4 = DoubleConv(64, 32, 1)
        self.down1 = Down(64, 128)
        # self.down2 = Down(128, 128)
        self.aspp = ASPP(128, 128)
        self.up1 = Up(256, 64)
        self.up2 = Up(128, 32)
        self.Att = Attention_block(F_g=32,F_l=32,F_int=16)
        self.outc = OutConv(32, n_classes)


    def forward(self, x):
        x1 = self.inc1(x)
        x_ = self.inc4(self.inc3(self.inc2(x1)))
        x2 = self.down1(x1)
        # x3 = self.down2(x2)
        x3 = self.aspp(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = self.Att(x, x_)
        logits = self.outc(x)
        # logits = self.outc(torch.cat([x_, x * x_], dim=1))
        return logits
