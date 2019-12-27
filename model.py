import torch
import torch.nn as nn
from collections import OrderedDict


class myModel(nn.Module):
    def __init__(self, n_channel, n_class):
        super().__init__()
        self.n_class = n_class
        self.NumOfMaxVar = 3
        self.bandExtract = nn.Sequential(OrderedDict([
            ('ext1', nn.Conv2d(n_channel, 16, kernel_size=1)), # in_channels, out_channels, kernel_size
            ('ext_bn1', nn.BatchNorm2d(16)),
            ('x1', nn.Sigmoid()),
            ('ext2', nn.Conv2d(16, 16, kernel_size=1)),
            ('ext_bn2', nn.BatchNorm2d(16)),
            ('x2', nn.Sigmoid())
            ]))
        self.featureSelcet = self.selcet
        self.conv1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 8, kernel_size=3, padding=1)),
            ('act1', nn.Sigmoid()),
            ('pool1', nn.MaxPool2d(kernel_size=2, stride=2))
        ]))
        self.conv2 = nn.Sequential(OrderedDict([
            ('conv2', nn.Conv2d(8, 16, kernel_size=3, padding=1)),
            ('act2', nn.Sigmoid()),
            ('pool2', nn.MaxPool2d(kernel_size=2, stride=2)),
            ]))
        # self.deconv1 = nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1),
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.bn1 = nn.BatchNorm2d(16)
        # self.deconv2 = nn.ConvTranspose2d(8, 3, kernel_size=3, stride=2, padding=1),
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.bn2 = nn.BatchNorm2d(8)
        self.conv1k = nn.Conv2d(8+16, n_class, 1)
        # self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax2d()

    def selcet(self, x):
        N, C, W, H = x.shape
        x0 = torch.zeros((N, self.NumOfMaxVar, W, H))
        # print(x.shape)
        for i, img in enumerate(x):
            vari = torch.zeros((C))
            for j, feature in enumerate(img):
                vari[j] = torch.var(feature.flatten())
            x0[i] = img[torch.argsort(-vari)[:self.NumOfMaxVar], :, :]
        return x0

    def forward(self, x):
        x = self.bandExtract(x)

        # N, C, W, H = x.shape
        # x0 = torch.zeros((N, self.NumOfMaxVar, W, H))
        # # print(x.shape)
        # for i, img in enumerate(x):
        #     vari = torch.zeros((C))
        #     for j, feature in enumerate(img):
        #         vari[j] = torch.var(feature.flatten())
        #     x0[i] = img[torch.argsort(-vari)[:self.NumOfMaxVar], :, :]
        
        x0 = self.featureSelcet(x)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)

        x2 = self.bn1(self.relu(self.upsample1(x2)))     
        x1 = self.bn2(self.relu(self.upsample2(x1)))  
        merge = torch.cat([x1, x2], dim=1)
        out = self.conv1k(merge)
        # out = self.softmax(out)

        return out


if __name__ == "__main__":
    net = myModel(10, 2)
    net1 = net.featureSelcet
    print(net1)