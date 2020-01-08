from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import visdom

from BagData import test_dataloader, train_dataloader
from FCN import FCN8s, FCN16s, FCN32s, FCNs, VGGNet


def train(epo_num=10, show_vgg_params=False):

    # vis = visdom.Visdom()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #vgg_model = VGGNet(requires_grad=True, show_params=show_vgg_params)
    #fcn_model = FCNs(pretrained_net=vgg_model, n_class=2)
    fcn_model = torch.load('./checkpoints_unet/fcn_model_0.pt')
    fcn_model = fcn_model.to(device)
    fcn_model = fcn_model.float()
    # criterion = nn.BCELoss().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(fcn_model.parameters(), lr=1e-2, momentum=0.7)

    all_train_iter_loss = []
    all_test_iter_loss = []

    # start timing
    prev_time = datetime.now()
    for epo in range(epo_num):
        
        train_loss = 0
        all_recall = 0.
        all_precision = 0.
        fcn_model.train()
        for index, (bag, bag_msk) in enumerate(train_dataloader):
            # bag.shape is torch.Size([4, 3, 160, 160])
            # bag_msk.shape is torch.Size([4, 2, 160, 160])

            bag = bag.to(device)
            bag_msk = bag_msk.to(device)

            optimizer.zero_grad()
            output = fcn_model(bag.float())
            # output = torch.sigmoid(output) # output.shape is torch.Size([4, 2, 160, 160])
            loss = criterion(output, bag_msk)
            loss.backward()
            iter_loss = loss.item()
            all_train_iter_loss.append(iter_loss)
            train_loss += iter_loss
            optimizer.step()

            outputData = np.argmax(output.data, 1)
            correction = (bag_msk * outputData).sum()
            recall = correction.to(torch.float64) / bag_msk.data.sum()
            precision = correction.to(torch.float64) / outputData.sum()
            all_recall += recall
            all_precision += precision

            # output_np = output.cpu().detach().numpy().copy() # output_np.shape = (4, 2, 160, 160)  
            # output_np = np.argmin(output_np, axis=1)
            # bag_msk_np = bag_msk.cpu().detach().numpy().copy() # bag_msk_np.shape = (4, 2, 160, 160) 
            # bag_msk_np = np.argmin(bag_msk_np, axis=1)

            print('epoch {}, {:03d}/{},train loss is {:.2f}'.format(epo, index, len(train_dataloader), iter_loss), end="        ")
            print('recall: {:.2f}, precision: {:.2f}, f-score: {:.2f}'.format(
                recall, precision, 2*(recall*precision)/(recall+precision)))
        
        test_loss = 0
        all_recall_test = 0.
        all_precision_test = 0.
        fcn_model.eval()
        with torch.no_grad():
            for index, (bag, bag_msk) in enumerate(test_dataloader):

                bag = bag.to(device)
                bag_msk = bag_msk.to(device)

                optimizer.zero_grad()
                output = fcn_model(bag)
                output = torch.sigmoid(output) # output.shape is torch.Size([4, 2, 160, 160])
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
        
                print("loss: {:.2}".format(iter_loss), end="        ")
                print('recall: {:.2}, precision: {:.2}, f-score: {:.2f}'.format(
                    recall_test, precision_test, 2*(recall_test*precision_test)/(recall_test+precision_test)))

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        prev_time = cur_time

        print('epoch train loss = %f, epoch test loss = %f, %s'
                %(train_loss/len(train_dataloader), test_loss/len(test_dataloader), time_str))
        
        rec, pre = all_recall/len(train_dataloader), all_precision/len(train_dataloader)
        f1 = 2*rec*pre / (rec+pre)
        print('epoch train recall, precision, f-score = %.2f, %.2f, %.2f' %(rec, pre, f1))

        rec, pre = all_recall_test/len(test_dataloader), all_precision_test/len(test_dataloader)
        f1 = 2*rec*pre / (rec+pre)
        print('epoch test  recall, precision, f-score = %.2f, %.2f, %.2f' %(rec, pre, f1))
        print('time: %s'%(time_str))

        torch.save(fcn_model, 'checkpoints_unet/fcn_model_{}.pt'.format(epo))
        print('saveing checkpoints_unet/fcn_model_{}.pt'.format(epo))


if __name__ == "__main__":

    train(epo_num=100, show_vgg_params=False)
