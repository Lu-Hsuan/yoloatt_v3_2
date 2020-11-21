import numpy as np
import cv2
import os

def kernel_resize(img_,ker_map,shape_r,shape_c,k):
    line_w = 0
    color_d = np.array([0,0,255])
    alpha = 0.8
    beta = 0.6
    gamma = 0
    row = shape_r//k
    col = shape_c//k
    arr_i = np.zeros((shape_r+line_w*row,shape_c+line_w*col,3),dtype=np.uint8)
    arr_m = np.zeros((shape_r+line_w*row,shape_c+line_w*col,3),dtype=np.uint8)
    #print(arr_i.shape)
    for i in range(0,shape_r,k):
        for j in range(0,shape_c,k):
            arr_i[i+line_w*i//k:i+k+line_w*i//k,j+line_w*j//k:j+k+line_w*j//k,...] = img_[i:i+k,j:j+k,...]
            arr_m[i+line_w*i//k:i+k+line_w*i//k,j+line_w*j//k:j+k+line_w*j//k,...] = ker_map[i//k,j//k]*color_d
    
    #cv2.imshow('map',arr_i)
    #cv2.imshow('map_',arr_m)
    img_add = cv2.addWeighted(arr_i, alpha, arr_m, beta, gamma)
    return arr_i,arr_m,img_add
    #cv2.imshow('add',img_add)
    #cv2.waitKey()

def only_kernel_resize(ker_map,shape_r,shape_c,k):
    line_w = 0
    color_d = np.array([0,0,255])
    alpha = 0.8
    beta = 0.6
    gamma = 0
    row = shape_r//k
    col = shape_c//k
    #arr_i = np.zeros((shape_r+line_w*row,shape_c+line_w*col,3),dtype=np.uint8)
    arr_m = np.zeros((shape_r+line_w*row,shape_c+line_w*col),dtype=np.float32)
    #print(arr_i.shape)
    for i in range(0,shape_r,k):
        for j in range(0,shape_c,k):
    #        arr_i[i+line_w*i//k:i+k+line_w*i//k,j+line_w*j//k:j+k+line_w*j//k,...] = img_[i:i+k,j:j+k,...]
            arr_m[i+line_w*i//k:i+k+line_w*i//k,j+line_w*j//k:j+k+line_w*j//k] = ker_map[min(i//k,row-1),j//k]
    return arr_m

if __name__ == "__main__":
    shape_r = 224
    shape_c = 320
    k = 32
    ker_map = np.random.normal(size=(shape_r//k,shape_c//k))
    img_i = np.random.normal(size=(shape_r,shape_c,3))
    kernel_resize(img_i,ker_map,shape_r,shape_c,k)
