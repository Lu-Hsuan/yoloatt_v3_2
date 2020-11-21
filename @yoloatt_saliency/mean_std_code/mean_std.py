import numpy as np
import cv2
import os

def kernel_cal(map_i,name,kernel,shape_r,shape_c):
    global phase,map_mean_p
    row = shape_r//kernel
    col = shape_c//kernel
    mean_th_L = 0.005
    std_th_L  = 0.005
    mean_th_H = 0.9
    std_th_H  = 0.9
    arr = np.zeros((2,row,col),dtype=np.float32)
    #print(arr.shape)
    map_std = map_i**2
    for i in range(0,shape_r,kernel):
        for j in range(0,shape_c,kernel):
    #        print(i,j)
            map_std_r = map_std[i:i+kernel,j:j+kernel]
            map_mea_r = map_i[i:i+kernel,j:j+kernel]
            #print(map_std_r.shape,map_mea_r.shape)
            mean = np.mean(map_mea_r)
            std  = np.sqrt(np.mean(map_std_r) - (mean**2))
            #std_ = np.sqrt(np.mean((map_mea_r - mean)**2))
            #print(std,std_)
    #        print(mean,std)
            arr[0,i//kernel,j//kernel] = mean
            arr[1,i//kernel,j//kernel] = std

    arr[0] = arr[0]/np.max(arr[0])
    arr[1] = arr[1]/np.max(arr[1])

    arr[0,arr[0]<mean_th_L] = 0 # mean < th = 0
    arr[1,arr[1]<std_th_L] = 0  # std < th = 0
    arr[0,arr[0]>mean_th_H] = 1 # mean > th = 1
    arr[1,arr[1]>std_th_H] = 1  # std > th = 1
    arr[1,arr[0]==0] = 0        # mean==0 std = 0

    np.save(f'{map_mean_p}/{phase}/{kernel}/{name}.npy',arr)
    return 

phase = 'train'
map_mean_p = f'map_mean'
shape_r,shape_c = 224,320
kernel = [8,16,32]
if(os.path.exists(map_mean_p)==False):
        os.mkdir(map_mean_p)
if(os.path.exists(f'{map_mean_p}/{phase}')==False):
        os.mkdir(f'{map_mean_p}/{phase}')
for k in kernel:
    if(os.path.exists(f'{map_mean_p}/{phase}/{k}')==False):
        os.mkdir(f'{map_mean_p}/{phase}/{k}')
    

file_ = os.listdir(f'{phase}')
data_num = len(file_)
print(data_num)
i = 0
for map_file in file_:
    name = map_file.split('.')[0]
    print(name,map_file)
    map_i = cv2.imread(f'{phase}/{map_file}',0)
    map_i = cv2.resize(map_i,(shape_c,shape_r),cv2.INTER_LINEAR)
    map_i = map_i/255.
    #print(np.sum(map_i))
    #print(map_i.shape)
    for k in kernel:
        #k = 1000
        #map_i = np.random.normal(5,10,(1000,1000))
        kernel_cal(map_i,name,k,map_i.shape[0],map_i.shape[1])
    i += 1
    print(i)
    #cv2.imshow('w',map_i)
    #cv2.waitKey()