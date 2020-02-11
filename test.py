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

from BagData import all_dataloader, test_dataloader
from model import myModel

# from unet import UNet

# %%
senceList = ["Barren", "Forest", "Grass/Crops","Shrubland", "Snow/Ice", "Urban", "Water", "Wetlands"]
def read_list(path='./dataLoad/result.txt'):
    f = open(path, "r")
    lines = f.readlines()
    senceDict = {}
    for i, line in enumerate(lines):
        senceId = re.split('[./]', line)[-3]
        senceDict[senceId] = i//12
    return senceDict

# %%
def test(modelPath):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # net = myModel(n_channel=10, n_class=2)
    # net = torch.load("./checkpoints_unet/unet_9.pt")
    net = torch.load("./checkpoints_attention/unet_attention_1.pt")
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

    senceDict = read_list()
    evaluateList = np.zeros((8, 5))
    for epo in range(1):
        train_loss = 0
        acc = 0.
        evaluateArray = np.zeros((5))
        # net.train()
        for index, (names, bag, bag_msk) in enumerate(test_dataloader):
            # bag.shape is torch.Size([4, 10, 512, 512])
            # bag_msk.shape is torch.Size([4, 2, 512, 512])

            bag = bag.to(device)
            bag_msk = bag_msk.to(device)

            output = net(bag)
            # # output = torch.sigmoid(output) # output.shape is torch.Size([4, 2, 160, 160])
            # regularization_loss = 0
            # # for param in net.parameters():
            # #     regularization_loss += torch.sum(torch.abs(param))
            # loss = criterion(output, bag_msk)
            
            # optimizer.zero_grad()
            # # loss.backward()
            # # optimizer.step()
            # iter_loss = loss.item()
            # all_train_iter_loss.append(iter_loss)
            # train_loss += iter_loss
            
            # print(bag_msk.shape, output.shape, torch.argmax(output, dim=1).shape)
            # correction = np.sum(bag_msk * np.argmax(output.detach(), 1))


            outputData = np.argmax(output.data, 1)
            evaluateArray[0] += bag_msk.data.sum()
            evaluateArray[1] += outputData.sum()
            evaluateArray[2] += (bag_msk * outputData).sum()
            # print(correction, bag_msk.data.sum())
            acc += (bag_msk == outputData).sum().to(torch.float64) / (256*256)
            # recall = correction.to(torch.float64) / bag_msk.data.sum()
            # precision = correction.to(torch.float64) / outputData.sum()
            recall = evaluateArray[2] / evaluateArray[0]
            precision = evaluateArray[2] / evaluateArray[1]

            print("{:03d}/{}, acc : {:.4f}, recall: {:.4f}, precision: {:.4f}, f-score: {:.4f}".format(index, len(test_dataloader), acc/(index + 1)/bag.shape[0], recall, precision, 2*(recall*precision)/(recall+precision)))

            for idx, name in enumerate(names):
                senceId = re.split('[_]', name)[0]
                out = output[idx]
                y_ = bag_msk[idx]
                y = np.argmax(out.data, 0)
                correction = (y_ * y).sum()
                sumY_ = y_.data.sum()
                sumY = y.sum()
                sumEqual = (y_ == y).sum()
                acc_ = sumEqual.to(torch.float64) / (256*256)
                # recall = correction / sumY_
                # precision = correction / sumY
                evaluateList[senceDict[senceId]] += np.array([1, acc_, correction, sumY_, sumY])
                

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        print('time: %s'%(time_str))

    print(evaluateList)
    np.save('./evaluateList.npy', evaluateList)
    for i in range(8):
        acc, recall, precision = evaluateList[i, 1] / evaluateList[i, 0], evaluateList[i, 2] / evaluateList[i, 3], evaluateList[i, 2] / evaluateList[i, 4]
        f1 = 2 * recall * precision / (recall + precision)
        print('{0} : acc : {1}, recall : {2}, precision : {3}, f1 : {4}'.format(senceList[i], acc, recall, precision, f1))
        

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--modelPath', dest='path', type=str, default="./checkpoints_unet/fcn_model_0.pt")
    args = parser.parse_args()
    test(args.path)
