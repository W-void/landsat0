import os
import re
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import argparse
from BagData import test_dataset
from torch.utils.data import DataLoader
import time

class MyTimer(object):       
    def __enter__(self):
        self.start = time.time()
        return self
        
    def __exit__(self, *unused):
        self.end = time.time()
        print("elapsed time: %f s" % (self.end-self.start))

test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=8)

def test(modelPath):
    xl, qal, ul, ml = [], [], [], []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    unet = torch.load("./checkpoints_unet/unet_1.pt")
    myModel = torch.load("./checkpoints_attention/aspp_4.pt")
    myModel = myModel.to(device).float()
    myModel.eval()
    unet = unet.to(device).float()
    unet.eval()

    for (names, bag, bag_msk, qa) in test_dataloader:
        bag = bag.to(device)
        with MyTimer():
            for i in range(10):
                m_output = myModel(bag)
        with MyTimer():
            for i in range(10):
                m_output = unet(bag)
        break



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--modelPath', dest='path', type=str, default="./checkpoints_unet/fcn_model_0.pt")
    args = parser.parse_args()

    test(args.path)
