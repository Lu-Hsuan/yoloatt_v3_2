import numpy as np
import cv2
import os
from uti import *
phase = 'val'
map_mean_p = f'test_w25_t'
img_p = f'../images/{phase}'
shape_r,shape_c = 224,320
kernel = [32,16,8]
sav_p = f'output_{map_mean_p}'
if(os.path.exists(sav_p)==False):
    os.mkdir(sav_p)

for i,k in enumerate(kernel):
    if(os.path.exists(f'{sav_p}/{i+1}')==False):
        os.mkdir(f'{sav_p}/{i+1}')

show = True
save = False
file_ = os.listdir(f'{map_mean_p}/{phase}/1')
data_num = len(file_)
print(data_num)
for npy_file in file_:
    name = npy_file.split('.')[0]
    print(name)
    img_i = cv2.imread(f'{img_p}/{name}.jpg')
    img_i = cv2.resize(img_i,(shape_c,shape_r),cv2.INTER_LINEAR)
    #cv2.namedWindow('img_i', 0)
    #cv2.imshow('img_i',img_i)
    for i,k in enumerate(kernel):
        k_file_p = f'{map_mean_p}/{phase}/{i+1}/{name}.npy'
        np_i = np.load(f'{k_file_p}')
        #print(np_i.shape)
        img_t,map_t,m_add_t = kernel_resize(img_i,np_i[0],shape_r,shape_c,k)
        img_t_,std_t,s_add_t = kernel_resize(img_i,np_i[1],shape_r,shape_c,k)
        print(map_t.shape)
        if show :
            cv2.namedWindow('img', 0),cv2.resizeWindow("img", shape_c*2, shape_r*2)
            cv2.namedWindow('mean', 0),cv2.resizeWindow("mean", shape_c*2, shape_r*2)
            cv2.namedWindow('std', 0),cv2.resizeWindow("std", shape_c*2, shape_r*2)
            cv2.namedWindow('mix_m', 0),cv2.resizeWindow("mix_m", shape_c*2, shape_r*2)
            #cv2.namedWindow('mix_s', 0),cv2.resizeWindow("mix_s", shape_c*2, shape_r*2)
            cv2.imshow('img',img_t)
            cv2.imshow('mean',map_t)
            cv2.imshow('std',std_t)
            cv2.imshow('mix_m',m_add_t)
            #cv2.imshow('mix_s',s_add_t)
            cv2.waitKey()
        if(save == True):
            img_total = np.concatenate([img_t,map_t,std_t,m_add_t],axis=-2)
            cv2.imwrite(f'{sav_p}/{i+1}/{name}.jpg',img_total)
    print()
