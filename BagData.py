import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from sklearn.preprocessing import OneHotEncoder
import cv2
import gdal
from gdalconst import *
from sklearn.preprocessing import OneHotEncoder


transform = transforms.Compose([
    transforms.ToTensor() # totensor 会改变shape！！！

    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class BagDataset(Dataset):

    def __init__(self, transform=None):
        self.transform = transform
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
        im_bands = dataset.RasterCount #波段数
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

train_size = int(0.9 * len(bag))
test_size = len(bag) - train_size
train_dataset, test_dataset = random_split(bag, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4)


if __name__ =='__main__':

    for train_batch in train_dataloader:
        print(train_batch)

    for test_batch in test_dataloader:
        print(test_batch)