# %%
import os
import re
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from focal_loss import FocalLoss
from BagData import test_dataloader, train_dataloader
from test import evaluate, showEvaluate, read_list
from unet import UNet, UNetWithAttention, UNetWithAttAndAspp, conv1, UNetWithGroupConv, SpoonNet, rendNet, SpoonNet2

# %%
def save_grad():
    def hook(grad):
        if torch.any(torch.isnan(grad)):
            print("grad is nan ...")
    return hook

def weights_init(m):
    classname=m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.xavier_uniform(m.weight.data)
        torch.nn.init.xavier_uniform(m.bias.data)

def load_checkpoint(net, net_pretrained=None):
    if net_pretrained == None:
        # net.apply(weights_init)
        return net
    else:
        net_dict = net.state_dict()
        net_pretrained_dict = net_pretrained.state_dict()
        pretrained_dict = {k: v for k, v in net_pretrained_dict.items() if k[:3] == 'inc'}
        # pretrained_dict = {k: v for k, v in net_pretrained_dict.items() if k in net_dict.keys()}
        # pretrained_dict.pop('outc.conv.weight')
        # pretrained_dict.pop('outc.conv.bias')
        print('Total : {}, update: {}'.format(len(net_pretrained_dict), len(pretrained_dict)))
        net_dict.update(pretrained_dict)
        net.load_state_dict(net_dict)
        for d, p in zip(net_dict, net.parameters()):
            if(d[:3] == 'inc'):
                p.requires_grad = False

        print("loaded finished!")
        return net

# %%
def train(epo_num=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # net = myModel(n_channel=10, n_class=2)
    net_pretrained = None
    # net_pretrained = torch.load("./checkpoints_attention/aspp_4.pt")
    # net_pretrained = torch.load("./checkpoints_attention/SpoonNet_5.pt")
    net = SpoonNet2(n_channels=10, n_classes=2, n_spectral=3)
    modelName = 'SpoonNetSpretral4'
    # net = UNet(10, 2)
    total_params = sum(p.numel() for p in net.parameters())
    print(total_params)
    net = load_checkpoint(net, net_pretrained)
    # print(net.state_dict().keys())
    net = net.to(device)
    net = net.float()
    # criterion = nn.BCELoss().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    # criterion = FocalLoss().to(device)
    # criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = optim.SGD(net.parameters(), lr=1e-1, momentum=0.9)
    # optimizer = optim.Adam(net.parameters(), lr=1e-1)

    all_train_iter_loss = []
    all_test_iter_loss = []
    writer = SummaryWriter('log')
    result = []
    global_step = 0
    # start timing
    prev_time = datetime.now()
    for epo in range(epo_num):
        if epo < 1:
            weights = [0.8, 0.2]
        elif epo < 2:
            weights = [0.2, 0.8]
        else:
            weights = [0, 1]
        train_loss = 0
        all_recall = 0.
        all_precision = 0.
        net.train()
        for index, (_, bag, bag_msk) in enumerate(train_dataloader):
            # bag.shape is torch.Size([4, 10, 512, 512])
            # bag_msk.shape is torch.Size([4, 2, 512, 512])

            bag = bag.to(device)
            bag_msk = bag_msk.to(device)

            [output, spectral] = net(bag)
            # output = net(bag)
            regularization_loss = 0
            # for param in net.parameters():
            #     regularization_loss += torch.sum(torch.abs(param))
            loss = weights[0] * criterion(spectral, bag_msk) + weights[1] * criterion(output, bag_msk)
            # loss = criterion(output, bag_msk)
            
            optimizer.zero_grad()
            # output.register_hook(print)
            # output.register_hook(save_grad())
            loss.backward()
            optimizer.step()
            iter_loss = loss.item()
            # writer.add_scalar('Loss/train_iter', iter_loss, global_step)
            # global_step += 1
            # # all_train_iter_loss.append(iter_loss)
            train_loss += iter_loss
           
            # # print(bag_msk.shape, output.shape, torch.argmax(output, dim=1).shape)
            # # correction = np.sum(bag_msk * np.argmax(output.detach(), 1))
            # outputData = np.argmax(output.data, 1)
            # correction = (bag_msk * outputData).sum()
            # # print(correction, bag_msk.data.sum())
            # recall = correction.to(torch.float64) / bag_msk.data.sum()
            # precision = correction.to(torch.float64) / outputData.sum()
            # all_recall += recall
            # all_precision += precision

            if np.mod(index, 15) == 0:
                print('epoch {}, {:03d}/{},train loss is {:.4f}'.format(epo, index, len(train_dataloader), iter_loss), end="\n        ")
            #     print('recall: {:.4f}, precision: {:.4f}, f-score: {:.4f}'.format(
            #         recall, precision, 2*(recall*precision)/(recall+precision)))

        if np.mod(epo+1, 1) == 0:
            savePath = './checkpoints_attention/'
            if not os.path.exists(savePath):
                os.makedirs(savePath)

            torch.save(net, savePath + modelName + '_{}.pt'.format(epo))
            print('saveing ' + savePath + modelName + '_{}.pt'.format(epo))

        test_loss = 0
        all_recall_test = 0.
        all_precision_test = 0.
        evaluateArray = np.zeros((4))
        senceDict = read_list()
        predEvalArray = np.zeros((8, 5))
        net.eval()
        with torch.no_grad():
            for index, (names, bag, bag_msk, _) in enumerate(test_dataloader):

                bag = bag.to(device)
                bag_msk = bag_msk.to(device)

                optimizer.zero_grad()
                [output, spectral] = net(bag)

                # loss = criterion(output, bag_msk)
                # iter_loss = loss.item()
                # # all_test_iter_loss.append(iter_loss)
                # test_loss += iter_loss

                outputData = np.argmax(output.data, 1)


                acc_test, recall_test, precision_test = get_acc_recall_precision(evaluateArray, bag_msk.data, outputData)

                if np.mod(index, 15) == 0:
                    print('epoch {}, {:03d}/{}'.format(epo, index, len(test_dataloader)), end="        ")
                    print('acc: {:.4}, recall: {:.4}, precision: {:.4}, f-score: {:.4f}'.format(acc_test/(index+1), 
                        recall_test, precision_test, 2*(recall_test*precision_test)/(recall_test+precision_test)))


                for idx, name in enumerate(names):
                    senceId = re.split('[_]', name)[0]
                    y = bag_msk.data[idx]
                    y_ = outputData[idx]
                    tmpList = evaluate(y, y_)
                    predEvalArray[senceDict[senceId]] += np.array(tmpList)


        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        prev_time = cur_time

        np.save('./log/' + modelName + '_{}.npy'.format(epo), predEvalArray)
        showEvaluate(predEvalArray)

        # train_loss = train_loss / len(train_dataloader)
        # test_loss = test_loss / len(test_dataloader)
        # print('epoch train loss = %f, epoch test loss = %f, %s'
        #         %(train_loss, test_loss, time_str))
        # writer.add_scalar('Loss/train', train_loss, epo)
        # writer.add_scalar('Loss/test', test_loss, epo)
        
        # rec, pre = all_recall/len(train_dataloader), all_precision/len(train_dataloader)
        # f1 = 2*rec*pre / (rec+pre)
        # print('epoch train recall, precision, f-score = %.4f, %.4f, %.4f' %(rec, pre, f1))

        print('epoch test acc: {:.4}, recall: {:.4}, precision: {:.4}, f-score: {:.4f}'.format(acc_test/(index+1), 
                        recall_test, precision_test, 2*(recall_test*precision_test)/(recall_test+precision_test)))
        
        print('time: %s'%(time_str))

        # result.append([test_loss, recall_test, precision_test])
        # np.save('./log/train_loss.npy', result)
        np.save('./log/train_eval_{}.npy'.format(epo), evaluateArray)
    
    writer.close()
    
def get_acc_recall_precision(arr, y, y_):
    arr[0] += y.sum()
    arr[1] += y_.sum()
    arr[2] += (y * y_).sum()
    arr[3] += (y == y_).sum().to(torch.float64) / (256*256) / y.shape[0]
    recall = arr[2] / arr[0]
    precision = arr[2] / arr[1]
    return arr[3], recall, precision


# %%
if __name__ == "__main__":
    train()
