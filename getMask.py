# %%
import os
import numpy as np
import torch
import torch.nn as nn
import cv2
import gdal
from gdalconst import *
from libtiff import TIFF


# %%
def readTif(fileName):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName+"文件无法打开")
        return
    im_width = dataset.RasterXSize #栅格矩阵的列数
    im_height = dataset.RasterYSize #栅格矩阵的行数
    im_bands = dataset.RasterCount #波段数
    im_data = dataset.ReadAsArray(0,0,im_width,im_height)
    return im_data


def getMask(imgIdx, checkPath='./checkpoints2/'):
    net = torch.load(checkPath + 'net15.pt')
    net.eval()

    imgPath = './VOC2012/JPEGImages/'
    maskPath = './VOC2012/SegmentationClass/'

    img = readTif(imgPath + '%05d'%imgIdx + '.tiff')[None, :, :, :]
    mask = net(torch.from_numpy(img*2e-5).float())

    GT = cv2.imread(maskPath + '%05d'%imgIdx + '.png', 0)
    return img[0, 1:4], mask[0], GT


# %%
if __name__ == "__main__":
    imgIdx = 1678
    img, mask, GT = getMask(imgIdx)
    print(GT)
    qa = cv2.imread('D:/Data/BC/image_qa/' + '%05d'%imgIdx + '.png', 0)
    # print(mask)
    cv2.imshow('color', img.transpose(1, 2, 0) * 2e-5)
    cv2.imshow('mask', np.float32(np.where(mask[0]>mask[1], 0, 1)))
    # cv2.imshow('GT', np.float32(np.where(GT==3, 1, 0)))
    cv2.imshow('GT', GT)
    cv2.imshow('QA', np.float32(qa))
    cv2.waitKey(0)
    cv2.destroyAllWindows()