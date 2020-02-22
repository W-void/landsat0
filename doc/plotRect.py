import cv2

names = ['color', 'mask', 'my', 'unet']
for name in names:
    img = cv2.imread('./log/LC801_'+name+'.jpg')
    img1 = img.copy()
    # print(img.shape)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    point1, point2 = (10, 80), (50, 120)
    cv2.rectangle(img1, point1, point2, (0, 255, 255), 2)
    cv2.namedWindow('img', 0)
    cv2.imshow('img', img1)
    cv2.waitKey(0)
    cv2.imwrite('./log/tmp_'+name+'.jpg', img1)

    min_x = min(point1[0],point2[0])     
    min_y = min(point1[1],point2[1])
    width = abs(point1[0] - point2[0])
    height = abs(point1[1] -point2[1])
    cut_img = img[min_y:min_y+height, min_x:min_x+width]
    cv2.imwrite('./log/tmp_cut_'+name+'.jpg', cut_img)