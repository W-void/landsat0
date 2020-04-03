import matplotlib.pyplot as plt
from sklearn import linear_model
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
from sklearn.metrics import mean_squared_error, r2_score

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

    nameList = os.listdir('/Users/wangshuli/Documents/BC/BC')
    nameList = [name for name in nameList if len(name) == 21]
    values = np.zeros((len(nameList)))
    qa_dict = dict(zip(nameList, values))
    m_dict = qa_dict.copy()
    u_dict = qa_dict.copy()
    x_dict = qa_dict.copy()

    for index, (names, bag, bag_msk, qa) in enumerate(test_dataloader):
        
        names = [name.split('_')[0] for name in names]
        bag = bag.to(device)
        m_output = myModel(bag)
        m_outputData = np.argmax(m_output.data, 1)
        u_output = unet(bag)
        u_outputData = np.argmax(u_output.data, 1)

        # for i, name in enumerate(names):
        #     x_dict[name] += bag_msk[i].sum().item() / (256*256)
        #     qa_dict[name] += qa[i].sum().item() / (256*256)
        #     m_dict[name] += m_outputData[i].sum().item() / (256*256)
        #     u_dict[name] += u_outputData[i].sum().item() / (256*256)

        for msk in bag_msk:
            xl.append(msk.sum().item() / (256*256))
        for pred in qa:
            qal.append(pred.sum().item() / (256*256))
        for pred in m_outputData:
            ml.append(pred.sum().item() / (256*256))
        for pred in u_outputData:
            ul.append(pred.sum().item() / (256*256))

        print(index)
        # if(index >= 20):
        #     break
    
    np.save(savePath, [xl, qal, ml, ul])


def drawLine(i):
    data = np.load(savePath)
    xl, yl = data[0], data[i]
    #通过x和y来建立线性模型
    xl, yl = np.array(xl)[:, None], np.array(yl)[:, None]
    linear = linear_model.LinearRegression()
    linear.fit(xl, yl)
    #查看模型系数β2
    print(linear.coef_) 
    #查看模型的截距β1
    print(linear.intercept_) 
    x_test = np.linspace(0, 1)[:, None]
    y_test = linear.predict(x_test)

    print('Coefficient of determination: %.4f'% r2_score(linear.predict(xl), yl))
    print('Mean squared error: %.4f'% mean_squared_error(linear.predict(xl), yl))

    fig, ax = plt.subplots()
    plt.scatter(xl, yl, marker='.',c='b',edgecolors='b')
    plt.plot(x_test, y_test, 'k')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim([0, 1.02])
    ax.set_ylim([0, 1.02])
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--modelPath', dest='path', type=str, default="./checkpoints_unet/fcn_model_0.pt")
    args = parser.parse_args()

    savePath = './log/x_list.npy'
    test(args.path)
    data = np.load(savePath)
    for i in range(1, 4):
        drawLine(i)