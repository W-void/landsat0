import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet_parts import MedianPool2d

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super(ASPP, self).__init__()
        
        
        mid_channels = int(out_channels / 2)

        self.conv_1x1_1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(mid_channels)

        self.conv_3x3_1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=2, dilation=2, groups=groups)
        self.bn_conv_3x3_1 = nn.BatchNorm2d(mid_channels)

        self.conv_3x3_2 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=5, dilation=5, groups=groups)
        self.bn_conv_3x3_2 = nn.BatchNorm2d(mid_channels)

        self.conv_3x3_3 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=groups)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(mid_channels)

        # self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_1x1_2 = nn.Conv2d(mid_channels * 4, out_channels, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(out_channels)

        # self.conv_1x1_3 = nn.Conv2d(5*mid_channels, out_channels, kernel_size=1) # (1280 = 5*256)
        # self.bn_conv_1x1_3 = nn.BatchNorm2d(out_channels)

        # self.conv_1x1_4 = nn.Conv2d(out_channels, num_classes, kernel_size=1)

    def forward(self, feature_map):
        # (feature_map has shape (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet instead is ResNet18_OS8 or ResNet34_OS8, it will be (batch_size, 512, h/8, w/8))

        # feature_map = self.maxPool(feature_map)
        feature_map_h = feature_map.size()[2] # (== h/16)
        feature_map_w = feature_map.size()[3] # (== w/16)

        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map))) # (shape: (batch_size, 256, h/16, w/16))

        # out_img = self.avg_pool(feature_map) # (shape: (batch_size, 512, 1, 1))
        # out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img))) # (shape: (batch_size, 256, 1, 1))
        # out_img = F.upsample(out_img, size=(feature_map_h, feature_map_w), mode="bilinear") # (shape: (batch_size, 256, h/16, w/16))

        # out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1) # (shape: (batch_size, 1280, h/16, w/16))
        # out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out))) # (shape: (batch_size, 256, h/16, w/16))
        # out = self.conv_1x1_4(out) # (shape: (batch_size, num_classes, h/16, w/16))
        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3], 1)
        out = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out)))
        # out = self.conv_1x1_2(out)
        return out

class DoubleAspp(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, kernelSize=3):
        super().__init__()
        self.double_aspp = nn.Sequential(
            nn.MaxPool2d(2),
            ASPP(in_channels, out_channels),
            ASPP(out_channels, out_channels)
        )

    def forward(self, x):
        return self.double_aspp(x)


class SingleAspp(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, kernelSize=3):
        super().__init__()
        self.single_aspp = nn.Sequential(
            nn.MaxPool2d(2),
            ASPP(in_channels, out_channels)
        )

    def forward(self, x):
        return self.single_aspp(x)