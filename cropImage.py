# %%
# -*- coding: utf-8 -*
import os
from glob import glob
import re
import cv2
import numpy as np
from libtiff import TIFF
import gdal
from gdalconst import *
import random


# %%
def make_txt(root='./VOC2012'):
    imgs = os.listdir(os.path.join(root, 'JPEGImages'))
    labels = os.listdir(os.path.join(root, 'SegmentationClass'))
    assert len(imgs) == len(labels)
    ratio = 0.8
    # train = imgs[:int(len(imgs) * ratio)]
    # val = imgs[int(len(imgs) * ratio):]
    train = np.random.choice(imgs, int(len(imgs) * ratio), replace=False)
    val = list(set(imgs) - set(train))
    txt_fname = root + '/ImageSets/Segmentation/' + 'train.txt'
    with open(txt_fname, 'a') as f:
        for t in train:
            f.writelines(t)
    txt_fname = root + '/ImageSets/Segmentation/' + 'val.txt'
    with open(txt_fname, 'a') as f:
        for v in val:
            f.writelines(v)


def read_images(root='./VOC2012', train=True):
    txt_fname = root + '/ImageSets/Segmentation/' + ('train.txt' if train else 'val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    data = [os.path.join(root, 'JPEGImages', i+'.jpg') for i in images]
    label = [os.path.join(root, 'SegmentationClass', i+'.png') for i in images]
    return data, label


# %%
def write_images(bands, path):
    img_width = bands.shape[2]
    img_height = bands.shape[1]
    num_bands = bands.shape[0]
    datatype = gdal.GDT_UInt16

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, img_width, img_height, num_bands, datatype)

    for i in range(num_bands):
        dataset.GetRasterBand(i + 1).WriteArray(bands[i])
    # print("save image success.")


def crop_img(root='../../Data/BC/', window_size=256, crop_method='random'):
    # sences = os.listdir(root)
    # sences = [i for i in sences if len(i) == len('LC80060102014147LGN00')]
    sences = ['LC80460282014171LGN00']
    # sences = ['LC82171112014297LGN00', 'LC81080182014238LGN00']
    valid_ext = ['.tif', '.TIF']
    num = 0
    trainOrVal = ['train', 'val']
    for tr in trainOrVal:
        if  not os.path.exists(os.path.join(root, tr, 'image')):
            os.makedirs(os.path.join(root, tr, 'image'))
        if  not os.path.exists(os.path.join(root, tr, 'label')):
            os.makedirs(os.path.join(root, tr, 'label'))
        if  not os.path.exists(os.path.join(root, tr, 'image_qa')):
            os.makedirs(os.path.join(root, tr, 'image_qa'))

    for j, sence in enumerate(sences):
        # print(sence)
        tifs = os.listdir(root + sence)
        tifs = [os.path.join(root, sence, tif) for tif in tifs if os.path.splitext(tif)[-1] in valid_ext]
        # tifs = ['B1', 'B10', 'B11', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'BQA', 'mask']
        data = []
        cols = []
        print("start read")
        tifs.sort()
        maskTif = tifs[-1]
        if maskTif[-8:] != 'mask.tif':
            print('...there is no mask tif')
        bandTifs = tifs[:11]
        qaTif = tifs[11]
        # bandTifs = ['B1', 'B10', 'B11', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9']
        print(bandTifs)
        
        # read mask
        maskImg = glob(root+sence+'/*.img')
        mask = gdal.Open(maskImg[0]).ReadAsArray()
        # mask = TIFF.open(maskTif).read_image()
        M, N = mask.shape

        # read QA
        qa = TIFF.open(qaTif).read_image()
        QA = np.where((qa & (3 << 14) == (3 << 14)) | (qa & (3 << 12) == (3 << 12)), 255, 128)
        QA = np.where((qa & (3 << 14) == (2 << 14)) | (qa & (3 << 12) == (2 << 12)), 192, QA)
        
        # read bands, 特别耗时
        valid_band = [0, *range(3, 9), 10, 1, 2]
        num_of_bands = len(valid_band)
        bands = np.zeros((num_of_bands, M, N), np.uint16)
        for i, band in enumerate(valid_band):
            print(bandTifs[band])
            Tif = TIFF.open(bandTifs[band])
            # if band == 9:
            #     bands[i] = cv2.resize(Tif.read_image(), (N, M), interpolation=cv2.INTER_NEAREST)
            #     continue  
            print(band)
            bands[i] = Tif.read_image()

        print("get bands")
        # fill, shadow, land, thinCloud, cloud = [0, 64, 128, 192, 255]

        if crop_method == 'random':
            iters = 400
            offset = 2400
            for i in range(iters):
                while True:
                    x, y = np.random.randint(0, min(M, N), size=2)
                    label = mask[x:x+window_size, y:y+window_size]
                    if np.sum(label == 0) == 0:
                        break
                img = bands[:, x:x+window_size, y:y+window_size]
                qa = QA[x:x+window_size, y:y+window_size]

                num = offset+i+iters*j
                tr =  trainOrVal[random.random() < 0.4]
                write_images(img, os.path.join(root, tr, 'image', sence + '_%05d.tiff'%(num)))
                cv2.imwrite(os.path.join(root, tr, 'label', sence + '_%05d.png'%(num)), np.uint8(label))
                cv2.imwrite(os.path.join(root, tr, 'image_qa', sence + '_%05d.png'%(num)), np.uint8(qa))

        elif crop_method == 'uniform':
            maxI, maxJ = M // window_size, N // window_size
            for idxI in range(maxI):
                iStart, iEnd = idxI*window_size, (idxI+1)*window_size
                for idxJ in range(maxJ):
                    jStart, jEnd = idxJ*window_size, (idxJ+1)*window_size
                    img = bands[:, iStart:iEnd, jStart:jEnd]
                    img = np.where(img < 0, 0, img)
                    # img = np.clip(img, a_min=0, a_max=5000)
                    label = mask[iStart:iEnd, jStart:jEnd]
                    if np.sum(label == 0) > 0:
                        continue
                    qa = QA[iStart:iEnd, jStart:jEnd]

                    tr =  trainOrVal[random.random() < 0.4]
                    write_images(img, os.path.join(root, tr, 'image', sence + '_%05d.tiff'%(num)))
                    cv2.imwrite(os.path.join(root, tr, 'label', sence + '_%05d.png'%(num)), np.uint8(label))
                    cv2.imwrite(os.path.join(root, tr, 'image_qa', sence + '_%05d.png'%(num)), np.uint8(qa))
                    num += 1

# %%
if __name__ == "__main__":
    crop_img(root='../BC/BC/', crop_method='uniform')
    # crop_img(root='/Users/wangshuli/Downloads/BC/', window_size=512, crop_method='uniform')