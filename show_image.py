# %%
import os
import re
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import argparse

from BagData import test_dataloader
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
to_pil_image = transforms.ToPILImage()
def test(modelPath):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    unet = torch.load("./checkpoints_unet/unet_11.pt")
    myModel = torch.load("./checkpoints_attention/unet_attention_2.pt")
    total_params = sum(p.numel() for p in unet.parameters())
    print(total_params)
    total_params = sum(p.numel() for p in myModel.parameters())
    print(total_params)

    unet = unet.to(device).float()
    myModel = myModel.to(device).float()
    
    unet.eval()
    myModel.eval()

    senceDict = read_list()
    
    for epo in range(1):
        for index, (names, bag, bag_msk, qa) in enumerate(test_dataloader):
            print(names, senceDict[names[0].split('_')[0]])
            bag = bag.to(device)
            bag_msk = bag_msk.to(device).data
            qa = qa.to(device).data
            u_output = unet(bag)
            u_outputData = np.argmax(u_output.data, 1)
            m_output = myModel(bag)
            m_outputData = np.argmax(m_output.data, 1)
            eval(bag_msk, qa)
            eval(bag_msk, u_outputData)
            eval(bag_msk, m_outputData)
            
            cv2.imshow('qa', qa[0].float().numpy())
            cv2.imshow('unet', u_outputData[0].float().numpy())
            cv2.imshow('my', m_outputData[0].float().numpy())
            cv2.imshow('mask', bag_msk[0].float().numpy())
            cv2.imshow("color", np.transpose(bag[0, 1:4].numpy() * 5e-6, (1, 2, 0)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            # img = to_pil_image(bag[0, 1:4] * 2e-5)
            # img.show()
            # img = to_pil_image(qa[0].float())
            # img.show()
            # img = to_pil_image(u_outputData[0].float())
            # img.show()
            # img = to_pil_image(m_outputData[0].float())
            # img.show()

            
def eval(y, y_):
    arr = np.zeros((4))
    arr[0] = y.sum()
    arr[1] = y_.sum()
    arr[2] = (y * y_).sum()
    arr[3] = (y == y_).sum().to(torch.float64) / (256*256)
    recall = arr[2] / arr[0]
    precision = arr[2] / arr[1]
    f1 = 2 * recall * precision / (recall + precision)
    return print("acc: {:.2f}, recall : {:.2f}, precision: {:.2f}, f1: {:.2f}".format(arr[3], recall, precision, f1))


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--modelPath', dest='path', type=str, default="./checkpoints_unet/fcn_model_0.pt")
    args = parser.parse_args()
    test(args.path)