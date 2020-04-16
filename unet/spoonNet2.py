import torch.nn.functional as F
from .unet_parts import *
from .aspp import *

class Up2(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size=3, group_size=1):
        super().__init__()

        self.group_size = group_size
        self.out_channels = out_channels
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, group_size)
        self.conv = DoubleConv(out_channels * 2, out_channels, kernel_size, group_size)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.conv1(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = []
        for i in range(self.group_size):
            start = int(i * self.out_channels/ self.group_size)
            end = int((i+1) * self.out_channels/ self.group_size)
            x.append(x1[:, start:end])
            x.append(x2[:, start:end])
        x = torch.cat((x), dim=1)
        # x = torch.cat([x2, x1], dim=1)
        return [x1, self.conv(x)]

class Attention_block_groups(nn.Module):
    def __init__(self,F_g,F_l):
        super(Attention_block_groups, self).__init__()
        self.W_g = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(F_g, F_l, kernel_size=3, stride=1, padding=1,bias=True, groups=F_l),
            nn.Conv2d(F_l, F_l, kernel_size=3, stride=1, padding=1,bias=True, groups=F_l),
            nn.BatchNorm2d(F_l),
            nn.Sigmoid()
            )
        # self.W_x = nn.Sequential(
        #     nn.Conv2d(F_l, F_l, kernel_size=1,stride=1,padding=0,bias=True, groups=F_l),
        #     nn.BatchNorm2d(F_l)
        # ) 
    def forward(self,g,x):
        g = self.W_g(g)
        return g*x

class SpoonNet2(nn.Module):
    def __init__(self, n_channels, n_classes, n_spectral=3):
        super(SpoonNet2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        self.inc1 = DoubleConv(n_channels, 64, 1)
        # self.inc1 = DoubleConv(32, 64, 1)
        self.inc2 = DoubleConv(64, 128, 1)
        self.inc3 = DoubleConv(128, 128, 1)
        self.inc3 = DoubleConv(128, 64, 1)
        self.inc4 = DoubleConv(64, n_spectral, 1)
        self.down1 = Down(n_spectral, n_spectral*32, 3, n_spectral)
        self.down2 = Down(n_spectral*32, n_spectral*64, 3, n_spectral)
        self.up1 = Up2(n_spectral*64, n_spectral*32, 3, n_spectral)
        self.up2 = Up2(n_spectral*32, n_spectral, 1, n_spectral)
        # self.Att = Attention_block_groups(F_g=3*32,F_l=3)
        self.outc1 = OutConv(n_spectral, n_classes)
        self.outc2 = OutConv(n_spectral, n_classes)

    def forward(self, x):
        x1 = self.inc4(self.inc3(self.inc2(self.inc1(x))))
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        [_, x] = self.up1(x3, x2)
        [spatial, x] = self.up2(x, x1)
        # x = self.Att(x, x1)
        logits = self.outc1(x)
        return [logits, x1, spatial]


# class SpoonNet2(nn.Module):
    # def __init__(self, n_channels, n_classes, n_spectral=3):
    #     super(SpoonNet2, self).__init__()
    #     self.n_channels = n_channels
    #     self.n_classes = n_classes

    #     self.inc1 = DoubleConv(n_channels, 64, 1)
    #     # self.inc1 = DoubleConv(32, 64, 1)
    #     self.inc2 = DoubleConv(64, 128, 1)
    #     self.inc3 = DoubleConv(128, 128, 1)
    #     self.inc3 = DoubleConv(128, 64, 1)
    #     self.inc4 = DoubleConv(64, 3, 1)
    #     self.down1 = Down(3, 3*32, 3, 3)
    #     self.down2 = Down(3*32, 3*64, 3, 3)
    #     self.down3 = Down(3*64, 3*128, 3, 3)
    #     self.down4 = Down(3*128, 3*256, 3, 3)
    #     self.up4 = Up2(3*256, 3*128, 3, 3)
    #     self.up1 = Up2(3*128, 3*64, 3, 3)
    #     self.up2 = Up2(3*64, 3*32, 3, 3)
    #     self.up3 = Up2(3*32, 3, 1, 3)
    #     # self.Att = Attention_block_groups(F_g=3*32,F_l=3)
    #     self.outc1 = OutConv(3, n_classes)
    #     self.outc2 = OutConv(3, n_classes)

    # def forward(self, x):
    #     x1 = self.inc4(self.inc3(self.inc2(self.inc1(x))))
    #     x2 = self.down1(x1)
    #     x3 = self.down2(x2)
    #     x4 = self.down3(x3)
    #     x5 = self.down4(x3)
    #     x = self.up4(x5, x4)
    #     x = self.up1(x4, x3)
    #     x = self.up2(x3, x2)
    #     x = self.up3(x, x1)
    #     # x = self.Att(x, x1)
    #     logits = self.outc1(x)
    #     return [logits, x1]
