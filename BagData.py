import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from sklearn.preprocessing import OneHotEncoder
import cv2
from osgeo import gdal
# from gdalconst import *
from sklearn.preprocessing import OneHotEncoder
import multiprocessing  # 解决VSCode对多线程支持不好的问题
multiprocessing.set_start_method('spawn',True)


transform = transforms.Compose([
    transforms.ToTensor() # totensor 会改变shape！！！
    , transforms.Normalize(
        mean=[0.04654*2, 0.04435*2, 0.04013*2, 0.04112*2, 0.04776*2, 0.02371*2, 0.01906*2, 0.0038*2, 0.1909*2, 0.17607*2], 
        std=[1370*16e-10, 1414*16e-10, 1385*16e-10, 1488*16e-10, 1522*16e-10, 998*16e-10, 821*16e-10, 292*16e-10, 2561*16e-10, 2119*16e-10]
        )
])

class BagDataset(Dataset):

    def __init__(self, transform=None):
        self.transform = transform
        # self.imgPath = './VOC2012/image/'
        # self.maskPath = './VOC2012/label/'
        self.imgPath = './VOC2012/JPEGImages/'
        self.maskPath = './VOC2012/SegmentationClass/'
        self.imgFiles = os.listdir(self.imgPath)
        # self.maskFiles = os.listdir(self.maskPath)
        
    def __len__(self):
        return len(self.imgFiles)

    def readTif(self, fileName):
        dataset = gdal.Open(fileName)
        if dataset == None:
            print(fileName+"文件无法打开")
            return
        im_width = dataset.RasterXSize #栅格矩阵的列数
        im_height = dataset.RasterYSize #栅格矩阵的行数
        # im_bands = dataset.RasterCount #波段数
        im_data = dataset.ReadAsArray(0,0,im_width,im_height)
        return im_data

    def __getitem__(self, idx):
        img_name = '%05d'%idx
        # print(self.imgPath+img_name+'.tiff'+'\n')
        img = self.readTif(self.imgPath+img_name+'.tiff')
        label = cv2.imread(self.maskPath+img_name+'.png', 0) # 灰度图
        # label = label[None, :, :]
        # 调整
        # img = img[1:4]
        label = label > 128
        # label = torch.FloatTensor(label)
        #print(imgB.shape)
        if self.transform:
            img = self.transform(img.transpose(1,2,0) * 2e-5)   
        # print(img.shape, label.shape)
        img = img.float()
        label = torch.tensor(label, dtype=torch.long)
        # print(label.shape, img.shape)
        return img, label

bag = BagDataset(transform)

train_size = int(0.8 * len(bag))
test_size = len(bag) - train_size
train_dataset, test_dataset = random_split(bag, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4)


if __name__ =='__main__':
    all_dataloader = DataLoader(bag, batch_size=1, shuffle=False, num_workers=4)
    for i, batch in enumerate(all_dataloader):
        if torch.any(torch.isnan(batch[0])):
            print("NO.{} have nan !!!".format(i))

    # for i in range(2):
    #     for train_batch in train_dataloader:
    #         print(train_batch[0].shape)

    #     for test_batch in test_dataloader:
    #         print(test_batch[0].shape)