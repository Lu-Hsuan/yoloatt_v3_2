import numpy as np
import cv2
import os

phase = 'train'
map_mean_p = f'map_mean'
shape_r,shape_c = 224,320
kernel = [8,16,32]
k = kernel[2]
k_file_p = f'{map_mean_p}/{phase}/{k}'
file_ = os.listdir(k_file_p)
data_num = len(file_)
print(data_num)
for np_file in file_:
    name = np_file.split('.')[0]
    print(name,np_file)
    np_i = np.load(f'{k_file_p}/{np_file}')
    print(np_i.shape)
    map_i = np_i[0]
    std_i = np_i[1]
    print(map_i.shape)
    map_i_t = cv2.imread(f'{phase}/{name}.png',0)
    map_i_t = cv2.resize(map_i_t,(map_i.shape[1],map_i.shape[0]),cv2.INTER_LINEAR)

    cv2.namedWindow('mean', 0)
    cv2.namedWindow('std', 0)
    cv2.namedWindow('map', 0)
    cv2.imshow('mean',map_i)
    cv2.imshow('std',std_i)
    cv2.imshow('map',map_i_t)
    cv2.waitKey()
