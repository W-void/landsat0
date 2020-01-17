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

from BagData import test_dataloader, train_dataloader
from model import myModel

from unet import UNet
from unet import  UNetWithAttention

def save_grad():
    def hook(grad):
        if torch.any(torch.isnan(grad)):
            print("grad is nan ...")
    return hook 
# %%
def save_grad():
    def hook(grad):
        if torch.any(torch.isnan(grad)):
            print("grad is nan ...")
    return hook

# %%
def train(epo_num=10):
    # vis = visdom.Visdom()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # net = myModel(n_channel=10, n_class=2)
    net = torch.load("./checkpoints_unet/unet_9.pt")
    # net = UNetWithAttention(n_channels=10, n_classes=2)
    total_params = sum(p.numel() for p in net.parameters())
    print(total_params)
    # print(net.state_dict().keys())
    net = net.to(device)
    net = net.float()
    # criterion = nn.BCELoss().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    # criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = optim.SGD(net.parameters(), lr=1e-2, momentum=0.7)

    all_train_iter_loss = []
    all_test_iter_loss = []

    # start timing
    prev_time = datetime.now()
    for epo in range(epo_num):
        
        train_loss = 0
        all_recall = 0.
        all_precision = 0.
        net.train()
        for index, (bag, bag_msk) in enumerate(train_dataloader):
            # bag.shape is torch.Size([4, 10, 512, 512])
            # bag_msk.shape is torch.Size([4, 2, 512, 512])

            bag = bag.to(device)
            bag_msk = bag_msk.to(device)

            output = net(bag)
            # output = torch.sigmoid(output) # output.shape is torch.Size([4, 2, 160, 160])
            regularization_loss = 0
            # for param in net.parameters():
            #     regularization_loss += torch.sum(torch.abs(param))
            loss = criterion(output, bag_msk) + 0.0001 * regularization_loss
            
            optimizer.zero_grad()
            # output.register_hook(print)
            # output.register_hook(save_grad())
            loss.backward()
            optimizer.step()
            iter_loss = loss.item()
            all_train_iter_loss.append(iter_loss)
            train_loss += iter_loss
           
            # print(bag_msk.shape, output.shape, torch.argmax(output, dim=1).shape)
            # correction = np.sum(bag_msk * np.argmax(output.detach(), 1))
            outputData = np.argmax(output.data, 1)
            correction = (bag_msk * outputData).sum()
            # print(correction, bag_msk.data.sum())
            recall = correction.to(torch.float64) / bag_msk.data.sum()
            precision = correction.to(torch.float64) / outputData.sum()
            all_recall += recall
            all_precision += precision

            # output_np = output.cpu().detach().numpy().copy() # output_np.shape = (4, 2, 160, 160)  
            # output_np = np.argmin(output_np, axis=1)
            # bag_msk_np = bag_msk.cpu().detach().numpy().copy() # bag_msk_np.shape = (4, 2, 160, 160) 
            # bag_msk_np =  np.argmin(bag_msk_np, axis=1)

            if np.mod(index, 15) == 14:
                print('epoch {}, {:03d}/{},train loss is {:.4f}'.format(epo, index, len(train_dataloader), iter_loss), end="        ")
                print('recall: {:.4f}, precision: {:.4f}, f-score: {:.4f}'.format(
                    recall, precision, 2*(recall*precision)/(recall+precision)))
                # # vis.close()
                # vis.images(output_np[:, None, :, :], win='train_pred', opts=dict(title='train prediction')) 
                # vis.images(bag_msk_np[:, None, :, :], win='train_label', opts=dict(title='label'))
                # vis.line(all_train_iter_loss, win='train_iter_loss',opts=dict(title='train iter loss'))

            # plt.subplot(1, 2, 1) 
            # plt.imshow(np.squeeze(bag_msk_np[0, ...]), 'gray')
            # plt.subplot(1, 2, 2) 
            # plt.imshow(np.squeeze(output_np[0, ...]), 'gray')
            # plt.pause(0.5)

        
        test_loss = 0
        all_recall_test = 0.
        all_precision_test = 0.
        net.eval()
        with torch.no_grad():
            for index, (bag, bag_msk) in enumerate(test_dataloader):

                bag = bag.to(device)
                bag_msk = bag_msk.to(device)

                optimizer.zero_grad()
                output = net(bag)
                # output = torch.sigmoid(output) # output.shape is torch.Size([4, 2, 160, 160])
                
                loss = criterion(output, bag_msk)
                iter_loss = loss.item()
                all_test_iter_loss.append(iter_loss)
                test_loss += iter_loss

                outputData = np.argmax(output.data, 1)
                correction = (bag_msk * outputData).sum()
                recall_test = correction.to(torch.float64) / bag_msk.data.sum()
                precision_test = correction.to(torch.float64) / outputData.sum()
                all_recall_test += recall_test
                all_precision_test += precision_test

                # output_np = output.cpu().detach().numpy().copy() # output_np.shape = (4, 2, 160, 160)  
                # output_np = np.argmin(output_np, axis=1)
                # bag_msk_np = bag_msk.cpu().detach().numpy().copy() # bag_msk_np.shape = (4, 2, 160, 160) 
                # bag_msk_np = np.argmin(bag_msk_np, axis=1)
        
                if np.mod(index, 15) == 0:
                    print("loss: {:.4}".format(iter_loss), end="        ")
                    print('recall: {:.4}, precision: {:.4}, f-score: {:.4f}'.format(
                        recall_test, precision_test, 2*(recall_test*precision_test)/(recall_test+precision_test)))
                    pass
                    # print(r'Testing... Open http://localhost:8097/ to see test result.')
                    # # vis.close()
                    # vis.images(output_np[:, None, :, :], win='test_pred', opts=dict(title='test prediction')) 
                    # vis.images(bag_msk_np[:, None, :, :], win='test_label', opts=dict(title='label'))
                    # vis.line(all_test_iter_loss, win='test_iter_loss', opts=dict(title='test iter loss'))


        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        prev_time = cur_time

        print('epoch train loss = %f, epoch test loss = %f, %s'
                %(train_loss/len(train_dataloader), test_loss/len(test_dataloader), time_str))
        
        rec, pre = all_recall/len(train_dataloader), all_precision/len(train_dataloader)
        f1 = 2*rec*pre / (rec+pre)
        print('epoch train recall, precision, f-score = %.4f, %.4f, %.4f' %(rec, pre, f1))

        rec, pre = all_recall_test/len(test_dataloader), all_precision_test/len(test_dataloader)
        f1 = 2*rec*pre / (rec+pre)
        print('epoch test  recall, precision, f-score = %.4f, %.4f, %.4f' %(rec, pre, f1))
        print('time: %s'%(time_str))
        
        if np.mod(epo+1, 1) == 0:
            savePath = './checkpoints_attention/'
            if not os.path.exists(savePath):
                os.makedirs(savePath)

            torch.save(net, savePath + 'unet_attention_{}.pt'.format(epo))
            print('saveing ' + savePath + 'unet_attention_{}.pt'.format(epo))


# %%
if __name__ == "__main__":
    train()
