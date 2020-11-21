import numpy as np
import cv2
import os
from uti import *
phase = 'val'
map_mean_p1 = f'mean_std_test'
map_mean_p2 = f'test_w25'
shape_r,shape_c = 224,320
kernel = [32,16,8]

file_ = os.listdir(f'{map_mean_p1}/{phase}/1')
data_num = len(file_)
print(data_num)
for npy_file in file_:
    name = npy_file.split('.')[0]
    print(name)
    for i,k in enumerate(kernel):
        k1_file_p = f'{map_mean_p1}/{phase}/{i+1}/{name}.npy'
        k2_file_p = f'{map_mean_p2}/{phase}/{i+1}/{name}.npy'
        np_i_1 = np.load(f'{k1_file_p}')
        np_i_2 = np.load(f'{k2_file_p}')
        print(np.all(np_i_1==np_i_2))
        #print(np_i.shape)
        
    print()
