# %%
import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
import visdom
import argparse

from BagData import test_dataloader

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
    net = torch.load("./checkpoints_attention/att_2.pt")
    # net = torch.load("./checkpoints_attention/unet_attention_2.pt")
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
    predEvalArray = np.zeros((8, 5))
    qaEvalArray = np.zeros((8, 5))
    
    for epo in range(1):
        train_loss = 0
        acc = 0.
        evaluateArray = np.zeros((4))
        qaArray = np.zeros((4))
        # net.train()
        for index, (names, bag, bag_msk, qa) in enumerate(test_dataloader):
            # bag.shape is torch.Size([4, 10, 512, 512])
            # bag_msk.shape is torch.Size([4, 2, 512, 512])

            bag = bag.to(device)
            bag_msk = bag_msk.to(device)
            # qa = qa.to(device)
            output = net(bag)
            outputData = np.argmax(output.data, 1)
           
            acc, recall, precision = get_acc_recall_precision(evaluateArray, bag_msk.data, outputData)
            # a, r, p = get_acc_recall_precision(qaArray, bag_msk.data, qa.data)

            if index % 10 == 0:
                print("{:03d}/{}, acc : {:.4f}, recall: {:.4f}, precision: {:.4f}, f-score: {:.4f}".format(index, len(test_dataloader), acc/(index + 1)/bag.shape[0], recall, precision, 2*(recall*precision)/(recall+precision)))
                # print("qa_mask, acc : {:.4f}, recall: {:.4f}, precision: {:.4f}, f-score: {:.4f}".format(a/(index + 1)/bag.shape[0], r, p, 2*(r*p)/(r+p)))

            for idx, name in enumerate(names):
                senceId = re.split('[_]', name)[0]
                out = output[idx]
                y = bag_msk[idx].data
                y_ = np.argmax(out.data, 0)
                tmpList = evaluate(y, y_)
                predEvalArray[senceDict[senceId]] += np.array(tmpList)
                # qa_ = qa[idx].data
                # tmpList = evaluate(y, qa_)
                # qaEvalArray[senceDict[senceId]] += np.array(tmpList)

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        print('time: %s'%(time_str))

    print(predEvalArray)
    np.save('./log/unetEvalArray.npy', predEvalArray)
    # np.save('./log/qaEvalArray.npy', qaEvalArray)
    showEvaluate(predEvalArray)
    # showEvaluate(qaEvalArray)
        

def get_acc_recall_precision(arr, y, y_):
    arr[0] += y.sum()
    arr[1] += y_.sum()
    arr[2] += (y * y_).sum()
    arr[3] += (y == y_).sum().to(torch.float64) / (256*256)
    recall = arr[2] / arr[0]
    precision = arr[2] / arr[1]
    return arr[3], recall, precision

def evaluate(y, y_):
    correction = (y_ * y).sum()
    sumY_ = y_.sum()
    sumY = y.sum()
    sumEqual = (y_ == y).sum()
    acc_ = sumEqual.to(torch.float64) / (256*256)
    return [1, acc_, correction, sumY, sumY_]

def showEvaluate(arr):
    for i in range(8):
        acc, recall, precision = arr[i, 1] / arr[i, 0], arr[i, 2] / arr[i, 3], arr[i, 2] / arr[i, 4]
        f1 = 2 * recall * precision / (recall + precision)
        print('{0} : acc : {1}, recall : {2}, precision : {3}, f1 : {4}'.format(senceList[i], acc, recall, precision, f1))


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--modelPath', dest='path', type=str, default="./checkpoints_unet/fcn_model_0.pt")
    args = parser.parse_args()
    test(args.path)
