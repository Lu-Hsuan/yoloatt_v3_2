from __future__ import division
import numpy as np
from cv2 import resize

def resize_fixation(img, shape_r=480, shape_c=640):
    out = np.zeros((shape_r, shape_c,1),dtype=np.uint8)
    factor_scale_r = shape_r / img.shape[0]
    factor_scale_c = shape_c / img.shape[1]

    coords = np.argwhere(img)
    for coord in coords:
        r = int(np.round(coord[0]*factor_scale_r))
        c = int(np.round(coord[1]*factor_scale_c))
        if r == shape_r:
            r -= 1
        if c == shape_c:
            c -= 1
        out[r,c,0] = 255

    return out

def preprocess_images(img,shape_r, shape_c):
    ims = np.zeros((shape_r, shape_c, 3))
    padded_image = resize(img, (shape_c, shape_r))
    ims = padded_image
    ims = ims[:, :, ::-1] ## BRG 2 RGB
    ims = np.moveaxis(ims, [0, 1, 2], [1, 2, 0]) #C H W
    #print(ims.shape)
    # ims = ims.transpose((0, 3, 1, 2))
    return ims


def preprocess_maps(img,shape_r, shape_c):
    ims = np.zeros((shape_r, shape_c, 1),dtype=np.uint8)
    padded_image = resize(img, (shape_c, shape_r))
    ims = padded_image[...,np.newaxis]
    ims = np.moveaxis(ims, [0, 1, 2], [1, 2, 0]) #C H W
    return ims

def preprocess_fixmaps(img,shape_r, shape_c):
    ims = np.zeros((shape_r, shape_c, 1),dtype=np.uint8)
    ims = resize_fixation(img, shape_r=shape_r, shape_c=shape_c)
    ims = np.moveaxis(ims, [0, 1, 2], [1, 2, 0]) #C H W
    return ims

