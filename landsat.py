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


# %%
def train(epo_num=50):
    # vis = visdom.Visdom()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = myModel(n_channel=10, n_class=2)
    net = net.float()
    net = net.to(device)
    # criterion = nn.BCELoss().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(net.parameters(), lr=1e-1, momentum=0.7)

    all_train_iter_loss = []
    all_test_iter_loss = []

    # start timing
    prev_time = datetime.now()
    for epo in range(epo_num):
        
        train_loss = 0
        net.train()
        for index, (bag, bag_msk) in enumerate(train_dataloader):
            # bag.shape is torch.Size([4, 10, 512, 512])
            # bag_msk.shape is torch.Size([4, 2, 512, 512])

            bag = bag.to(device)
            bag_msk = bag_msk.to(device)

            optimizer.zero_grad()
            output = net(bag)
            # output = torch.sigmoid(output) # output.shape is torch.Size([4, 2, 160, 160])
            loss = criterion(output, bag_msk)
            loss.backward()
            iter_loss = loss.item()
            all_train_iter_loss.append(iter_loss)
            train_loss += iter_loss
            optimizer.step()

            output_np = output.cpu().detach().numpy().copy() # output_np.shape = (4, 2, 160, 160)  
            output_np = np.argmin(output_np, axis=1)
            bag_msk_np = bag_msk.cpu().detach().numpy().copy() # bag_msk_np.shape = (4, 2, 160, 160) 
            bag_msk_np =  np.argmin(bag_msk_np, axis=1)

            if np.mod(index, 15) == 14:
                print('epoch {}, {}/{},train loss is {}'.format(epo, index, len(train_dataloader), iter_loss))
                
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
        net.eval()
        with torch.no_grad():
            for index, (bag, bag_msk) in enumerate(test_dataloader):

                bag = bag.to(device)
                bag_msk = bag_msk.to(device)

                optimizer.zero_grad()
                output = net(bag)
                output = torch.sigmoid(output) # output.shape is torch.Size([4, 2, 160, 160])
                loss = criterion(output, bag_msk)
                iter_loss = loss.item()
                print(iter_loss)
                all_test_iter_loss.append(iter_loss)
                test_loss += iter_loss

                output_np = output.cpu().detach().numpy().copy() # output_np.shape = (4, 2, 160, 160)  
                output_np = np.argmin(output_np, axis=1)
                bag_msk_np = bag_msk.cpu().detach().numpy().copy() # bag_msk_np.shape = (4, 2, 160, 160) 
                bag_msk_np = np.argmin(bag_msk_np, axis=1)
        
                if np.mod(index, 15) == 0:
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
        

        if np.mod(epo, 5) == 0:
            torch.save(net, './checkpoints2/net{}.pt'.format(epo))
            print('saveing checkpoints2/net{}.pt'.format(epo))


# %%
if __name__ == "__main__":
    train()
