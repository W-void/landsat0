import cv2
import os


def draw_rectangle(event,x,y,flags,param):
    global ix, iy
    if event==cv2.EVENT_LBUTTONDOWN:
        ix, iy = x, y
        print("point1:=", x, y)
    # elif event==cv2.EVENT_LBUTTONUP:
    #     print("point2:=", x, y)
    #     print("width=",x-ix)
    #     print("height=", y - iy)
    #     cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)

def plotRect(imgName):
    names = ['color', 'mask', 'spectral', 'my', 'unet']
    for name in names:
        img = cv2.imread(root+imgName+name+'.jpg')
        img1 = img.copy()
        # print(img.shape)
        # cv2.imshow('img', img)
        # cv2.waitKey(0）

        cv2.namedWindow('img')
        if name == 'color':
            cv2.setMouseCallback('img', draw_rectangle)
            while(1):
                cv2.imshow('img', img1)
                if cv2.waitKey(20) & 0xFF == 27:
                    break
        size = 20
        point1, point2 = (ix-size, iy-size), (ix+size, iy+size)
        cv2.rectangle(img1, point1, point2, (0, 255, 255), 2)
        cv2.imshow('img', img1)
        cv2.waitKey(0)
        min_x = min(point1[0],point2[0])     
        min_y = min(point1[1],point2[1])
        # width = abs(point1[0] - point2[0])
        # height = abs(point1[1] -point2[1])
        cut_img = img[min_y:min_y+2*size, min_x:min_x+2*size]
        cv2.destroyAllWindows()
        cv2.imwrite(root+'cut/'+ imgName+name +'.jpg', img1)
        cv2.imwrite(root+'cut/tmp_cut_'+imgName+name+'.jpg', cut_img*1.2)
        # size = 20
        # point1, point2 = (ix-size, iy-size), (ix+size, iy+size)
        # cv2.rectangle(img1, point1, point2, (0, 255, 255), 2)
        # cv2.imshow('img', img1)
        # min_x = min(point1[0],point2[0])     
        # min_y = min(point1[1],point2[1])
        # width = abs(point1[0] - point2[0])
        # height = abs(point1[1] -point2[1])
        # cut_img = img[min_y:min_y+height, min_x:min_x+width]
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.imwrite(root+'tmp_cut_'+imgName+name+'.jpg', cut_img)


if __name__ == '__main__':
    root = './log/spoon2/'
    jpgs = [j[:-9] for j in os.listdir(root) if j[-9:] == 'color.jpg']

    for jpg in jpgs:
        plotRect(jpg)

