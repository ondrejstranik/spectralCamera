'''
image warping with linear interpolation
this is just partial code for replacement of warp in skimage
 TODO: . Finish it

'''

#%% package

import numpy as np

# %%
def warp(img, src_x, src_y):
    sz = img.shape()
    dst = np.zeros(sz)
    src_xl = Floor(Int, srx_x)
    src_xrem = src_x - src_xl
    img[:] = (1-src_xrem)*img[src_xl]+ src_xrem*img[src_xl+1]

