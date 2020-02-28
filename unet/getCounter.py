import cv2
import os

pic = cv2.imread('./log/LC801_mask.jpg', 0)
ret, binary = cv2.threshold(pic,127,255,cv2.THRESH_BINARY) 
contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE) # contours[0]是边界的轮廓
cv2.drawContours(binary, contours, -1, (125), 1) # 原地改变
# for contour in range(contours[1:]):  # 如果不想要边界的轮廓
#     cv2.drawContours(binary, contour, -1, (125), 1)
plt.imshow(binary)
plt.show()