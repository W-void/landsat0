# %%
import os
import re
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
import visdom
from math import isnan
import argparse

from BagData import all_dataloader
from model import myModel

# from unet import UNet

# %%
def test(modelPath):
    # vis = visdom.Visdom()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # net = myModel(n_channel=10, n_class=2)
    net = torch.load(modelPath)
    total_params = sum(p.numel() for p in net.parameters())
    print(total_params)
    # net = UNet(n_channels=10, n_classes=2)
    # print(net.state_dict().keys())
    net = net.to(device)
    net = net.float()
    # criterion = nn.BCELoss().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    # criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = optim.SGD(net.parameters(), lr=1e-2, momentum=0.7)

    all_train_iter_loss = []
    all_test_iter_loss = []
    net.eval()
    # start timing
    prev_time = datetime.now()
    for epo in range(1):
        
        train_loss = 0
        acc = 0.
        evaluateArray = np.zeros((3))
        # net.train()
        for index, (bag, bag_msk) in enumerate(all_dataloader):
            # bag.shape is torch.Size([4, 10, 512, 512])
            # bag_msk.shape is torch.Size([4, 2, 512, 512])

            bag = bag.to(device)
            bag_msk = bag_msk.to(device)

            output = net(bag)
            # output = torch.sigmoid(output) # output.shape is torch.Size([4, 2, 160, 160])
            regularization_loss = 0
            # for param in net.parameters():
            #     regularization_loss += torch.sum(torch.abs(param))
            loss = criterion(output, bag_msk)
            
            optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            iter_loss = loss.item()
            all_train_iter_loss.append(iter_loss)
            train_loss += iter_loss
            
            # print(bag_msk.shape, output.shape, torch.argmax(output, dim=1).shape)
            # correction = np.sum(bag_msk * np.argmax(output.detach(), 1))
            outputData = np.argmax(output.data, 1)
            correction = (bag_msk * outputData).sum()
            evaluateArray[0] += bag_msk.data.sum()
            evaluateArray[1] += outputData.sum()
            evaluateArray[2] += (bag_msk * outputData).sum()
            # print(correction, bag_msk.data.sum())
            acc += (bag_msk == outputData).sum().to(torch.float64) / (256*256)
            # recall = correction.to(torch.float64) / bag_msk.data.sum()
            # precision = correction.to(torch.float64) / outputData.sum()
            recall = evaluateArray[2] / evaluateArray[0]
            precision = evaluateArray[2] / evaluateArray[1]
            print("{:03d}/{}, acc : {:.4f}, recall: {:.4f}, precision: {:.4f}, f-score: {:.4f}".format(index, len(all_dataloader), acc/(index + 1), recall, precision, 2*(recall*precision)/(recall+precision)))
            

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        print('time: %s'%(time_str))
        

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--modelPath', dest='path', type=str, default="./checkpoints_unet/fcn_model_0.pt")
    args = parser.parse_args()
    test(args.path)
