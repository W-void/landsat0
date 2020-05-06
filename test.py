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
from skimage import measure
from BagData import test_dataloader
from log.npy2tex import npy2tex
# from unet import UNet

from sklearn.metrics import roc_curve, auc

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
    # net = torch.load("./checkpoints_unet/unet_1.pt")
    # net = torch.load("./checkpoints_attention/aspp_4.pt")
    # net = torch.load("./checkpoints_attention/SpoonNetSpretral3_12.pt", map_location=torch.device('cpu'))
    # total_params = sum(p.numel() for p in net.parameters())
    # print(total_params)
    # # net = UNet(n_channels=10, n_classes=2)
    # # print(net.state_dict().keys())
    # net = net.to(device)
    # net = net.float()
    # net.eval()

    net2 = torch.load("./checkpoints_unet/unet_1.pt")
    total_params = sum(p.numel() for p in net2.parameters())
    print(total_params)
    net2 = net2.to(device)
    net2 = net2.float()
    net2.eval()

    net3 = torch.load("./checkpoints_attention/SegNet_2.pt")
    total_params = sum(p.numel() for p in net3.parameters())
    print(total_params)
    net3 = net3.to(device)
    net3 = net3.float()
    net3.eval()

    all_train_iter_loss = []
    all_test_iter_loss = []
    
    # start timing
    prev_time = datetime.now()

    senceDict = read_list()
    predEvalArray = np.zeros((8, 5))
    unetEvalArray = np.zeros((8, 5))
    segEvalArray = np.zeros((8, 5))
    qaEvalArray = np.zeros((8, 5))

    roc = np.zeros((2, 100))
    
    for epo in range(1):
        train_loss = 0
        acc = 0.
        evaluateArray = np.zeros((4))
        qaArray = np.zeros((4))
        # net.train()
        for index, (names, bag, bag_msk, qa) in enumerate(test_dataloader):
            # bag.shape is torch.Size([4, 10, 512, 512])
            # bag_msk.shape is torch.Size([4, 1, 512, 512])

            bag = bag.to(device)
            bag_msk = bag_msk.to(device)
            # qa = qa.to(device)
            # [output, spectral, _] = net(bag)
            # outputData = np.argmax(output.data, 1)

            output2 = net2(bag)
            outputData2 = np.argmax(output2.data, 1)

            output3 = net3(bag)
            outputData3 = np.argmax(output3.data, 1)
           
            regionSelect(bag_msk.data)
            # regionSelect(outputData)
            regionSelect(outputData2)
            regionSelect(outputData3)
            # regionSelect(qa.data)

            if index % 10 == 0:
                print(index)

            for idx, name in enumerate(names):
                senceId = re.split('[_]', name)[0]
                y = bag_msk.data[idx]
                # y_ = outputData[idx]
                # tmpList = evaluate(y, y_)
                # predEvalArray[senceDict[senceId]] += np.array(tmpList)

                y_ = outputData2[idx]
                tmpList = evaluate(y, y_)
                unetEvalArray[senceDict[senceId]] += np.array(tmpList)

                y_ = outputData3[idx]
                tmpList = evaluate(y, y_)
                segEvalArray[senceDict[senceId]] += np.array(tmpList)

                # qa_ = qa.data[idx]
                # tmpList = evaluate(y, qa_)
                # qaEvalArray[senceDict[senceId]] += np.array(tmpList)

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        print('time: %s'%(time_str))

    # print(predEvalArray)
    # np.save('./log/spoonNetEvalArray_region.npy', predEvalArray)
    np.save('./log/unetEvalArray_region.npy', unetEvalArray)
    np.save('./log/segEvalArray_region.npy', segEvalArray)
    # np.save('./log/qaEvalArray_region.npy', qaEvalArray)
    # showEvaluate(predEvalArray)
    # showEvaluate(unetEvalArray)
    # showEvaluate(segEvalArray)
    npy2tex(unetEvalArray)
    npy2tex(segEvalArray)
    # showEvaluate(qaEvalArray)
    # np.save('./log/roc.npy', roc)
    # AUC = auc(roc[1, 1:]/ roc[1, 0], roc[0, 1:]/ roc[0, 0])
    # print('AUC : ', AUC)


def regionSelect(datas):
    for i, data in enumerate(datas):
        labels = measure.label(data, connectivity=1)  #8连通区域标记
        #筛选连通区域大于30的
        properties = measure.regionprops(labels)
        valid_label = set()
        for prop in properties:
            if prop.area < 30:
                valid_label.add(prop.label)
        data = np.in1d(labels, list(valid_label)).reshape(labels.shape)
        datas[i] = torch.from_numpy(data)


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
    arr = np.sum(arr, 0)
    acc, recall, precision = arr[1] / arr[0], arr[2] / arr[3], arr[2] / arr[4]
    f1 = 2 * recall * precision / (recall + precision)
    print("total_acc: {}, total_f1: {}, total_recall: {}, total_precision: {}".format(acc, f1, recall, precision))


def get_roc(arr, mask, logit):
    mask, logit = mask.flatten(), logit.flatten()
    pos, neg = torch.sum(mask), torch.sum(1-mask)
    arr[0, 0] += pos
    arr[1, 0] += neg
    for thres in range(1, 100):
        tmp = torch.where(logit > thres / 100, torch.full_like(mask, 1), torch.full_like(mask, 0))
        TPR = torch.sum(tmp * mask)
        FPR = torch.sum(tmp * (1 - mask))
        arr[0, thres] += TPR
        arr[1, thres] += FPR

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--modelPath', dest='path', type=str, default="./checkpoints_unet/fcn_model_0.pt")
    args = parser.parse_args()
    test(args.path)
