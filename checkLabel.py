# %%
import os
import numpy as np
from libtiff import TIFF
import matplotlib.pyplot as plt


# %%
root = 'D:/Data/BC/'
sences = os.listdir(root)
sences = [i for i in sences if len(i) == len('LC80060102014147LGN00')]
valid_ext = ['.tif', '.TIF']

for j, sence in enumerate(sences):
    tifs = os.listdir(root + sence)
    tifs = [os.path.join(root, sence, tif) for tif in tifs if os.path.splitext(tif)[-1] in valid_ext]
    maskTif = tifs[-1]
    Tif = TIFF.open(maskTif)
    mask = Tif.read_image()
    uniq = np.unique(mask)
    print(sence, ' : ', uniq)
    plt.imshow(mask)
    # plt.show()