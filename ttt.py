import cv2
import numpy as np
from skimage import measure
import sys

sys.setrecursionlimit(1000000000)

def changeColor(x, y):
    if np.all(img[x, y] > 200):
        img[x, y] = 100
        changeColor(x+1, y)
        changeColor(x-1, y)
        changeColor(x, y+1)
        changeColor(x, y-1)

img = cv2.imread('./ttt.jpeg', 1)
print(img.shape)
data = np.where(img[:, :, 0] > 120, 1, 0)
labels = measure.label(data, connectivity=2)

gray = 150
label = labels[300, 500]
idx = np.where(labels == label)
img[idx[0], idx[1], :] = gray

label = labels[800, 400]
idx = np.where(labels == label)
img[idx[0], idx[1], :] = gray

label = labels[835, 620]
idx = np.where(labels == label)
img[idx[0], idx[1], :] = gray

mask = np.where((img[:,:,0]==gray) & (img[:,:,1]==gray) & (img[:,:,2]==gray), 1, 0)
# mask = 1-data
kernel = np.ones((3,3),np.uint8)
dilation = cv2.dilate(mask.astype(np.float32), kernel, iterations = 1)
tmp_mask = dilation - mask
idx = np.where(tmp_mask)
img[idx[0], idx[1], :] = [115, 67, 56]

cv2.namedWindow('img', 0)
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.imwrite('./tttt.jpg', img)



