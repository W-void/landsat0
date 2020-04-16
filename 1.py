import cv2

img = cv2.imread('./log/spoon2/LC80980712014024LGN00_15079_spectral.jpg', 1)
for i in range(3):
    img[:, :, i] = (img[:, :, i] -  img[:, :, i].min())/  (img[:, :, i].max() - img[:, :, i].min()) * 255
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

