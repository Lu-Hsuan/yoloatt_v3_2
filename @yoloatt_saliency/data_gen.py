import numpy as np
import cv2
from PIL import Image
import scipy.io
from pathlib import Path
import os
import random
import json
import copy
from utility import *
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class generator_SAL:
    def __init__(self,phase,shape_r,shape_c,batch_size,data_path='../train_backbone'):
        self.phase = phase
        self.shape_r = shape_r
        self.shape_c = shape_c
        self.batch_size = batch_size
        self.img_p = os.path.join(data_path, 'images')
        self.fix_p = os.path.join(data_path, 'fixations_map')
        self.map_p = os.path.join(data_path, 'maps')
        self.count = 0
        self.file_ = os.listdir(f'{self.img_p}/{self.phase}')
        self.data_num = len(self.file_)
        random.shuffle(self.file_)
        print(f'Dataset : {self.phase} number : {self.data_num}')
    def __iter__(self):
        while(self.count < self.data_num):
            img_ = []
            map_   = []
            fix_   = []
            file_batch = self.file_[self.count:min(self.data_num,self.count+self.batch_size)]
            for filename in file_batch:
                img_nr = filename.split('.')[0]
                #print(img_nr)
                img_i,map_i,fix_i = self.get_data(img_nr)
                img_.append(preprocess_images(img_i,self.shape_r,self.shape_c))
                map_.append(preprocess_maps(map_i,self.shape_r,self.shape_c))
                fix_.append(preprocess_fixmaps(fix_i,self.shape_r,self.shape_c))
            self.count += self.batch_size
            yield np.array(img_) , np.array(map_) , np.array(fix_)
        self.count = 0
        random.shuffle(self.file_)

    def get_data(self,img_nr):
        img_file = f'{self.img_p}/{self.phase}/{img_nr}.jpg'
        map_file = f'{self.map_p}/{self.phase}/{img_nr}.png'
        fix_file = f'{self.fix_p}/{self.phase}/{img_nr}.png'
        #print(img_file)
        if os.path.exists(img_file):
            img_i = cv2.imread(str(img_file))
            map_i = cv2.imread(str(map_file), cv2.IMREAD_GRAYSCALE)
            fix_i = cv2.imread(str(fix_file), cv2.IMREAD_GRAYSCALE)
            return img_i,np.expand_dims(map_i,axis=-1),np.expand_dims(fix_i,axis=-1)
        else:
            raise RuntimeError('img Wrong')
            return None

class generator_SAL_torch(Dataset):
    def __init__(self,phase,shape_r,shape_c):
        self.phase = phase
        self.shape_r = shape_r
        self.shape_c = shape_c
        #self.batch_size = batch_size
        self.img_p = 'images'
        self.fix_p = 'fixations_map'
        self.map_p = 'maps'
        self.count = 0
        self.file_ = os.listdir(f'{self.img_p}/{self.phase}')
        self.data_num = len(self.file_)
        self.transform_i = transforms.Compose([ 
                        transforms.ToPILImage(),
                        transforms.Resize((shape_r, shape_c),interpolation=cv2.INTER_LINEAR),
                        transforms.ToTensor(),
                        #transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        #                    std=[0.229, 0.224, 0.225])
                    ])
        self.transform_m = transforms.Compose([ 
                        transforms.ToPILImage(),
                        transforms.Resize((shape_r, shape_c),interpolation=cv2.INTER_LINEAR),
                        transforms.ToTensor()
                    ])
        #random.shuffle(self.file_)
        print(f'Dataset : {self.phase} number : {self.data_num}')
    def __getitem__(self, index):
        ##############################################
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform).
        # 3. Return the data (e.g. image and label)
        ##############################################
        img_nr = self.file_[index].split('.')[0]
        #print(img_nr)
        img_file = f'{self.img_p}/{self.phase}/{img_nr}.jpg'
        map_file = f'{self.map_p}/{self.phase}/{img_nr}.png'
        fix_file = f'{self.fix_p}/{self.phase}/{img_nr}.png'
        #print(img_file)
        img_i = cv2.imread(str(img_file))
        img_i = cv2.cvtColor(img_i,cv2.COLOR_BGR2RGB)
        map_i = cv2.imread(str(map_file), cv2.IMREAD_GRAYSCALE)
        fix_i = cv2.imread(str(fix_file), cv2.IMREAD_GRAYSCALE)

        map_i = np.expand_dims(map_i,axis=-1)
        fix_i = np.expand_dims(fix_i,axis=-1)

        img_i = self.transform_i(img_i)
        map_i = self.transform_m(map_i)
        fix_i = preprocess_fixmaps(fix_i,self.shape_r,self.shape_c)
        #print(img_i.size())
        return img_i,map_i,torch.from_numpy(fix_i).float()/255.
    def __len__(self):
        ##############################################
        ### Indicate the total size of the dataset
        ##############################################
        return self.data_num

class generator_SAL_metric(Dataset):
    def __init__(self,phase,shape_r,shape_c,tar_shape_r=480,tar_shape_c=640,data_path='../train_backbone'):
        self.phase = phase
        self.shape_r = shape_r
        self.shape_c = shape_c
        self.tar_shape_r = tar_shape_r
        self.tar_shape_c = tar_shape_c
        #self.batch_size = batch_size
        self.img_p = os.path.join(data_path, 'images')
        self.fix_p = os.path.join(data_path, 'fixations_map')
        self.map_p = os.path.join(data_path, 'maps')
        self.count = 0
        self.file_ = os.listdir(os.path.join(self.img_p,self.phase))
        self.data_num = len(self.file_)
        self.transform_i = transforms.Compose([ 
                        transforms.ToPILImage(),
                        transforms.Resize((shape_r, shape_c),interpolation=cv2.INTER_LINEAR),
                        transforms.ToTensor(),
                        #transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        #                    std=[0.229, 0.224, 0.225])
                    ])
        self.transform_m = transforms.Compose([ 
                        transforms.ToPILImage(),
                        transforms.Resize((tar_shape_r, tar_shape_c),interpolation=cv2.INTER_LINEAR),
                        transforms.ToTensor()
                    ])
        #random.shuffle(self.file_)
        print(f'Dataset : {self.phase} number : {self.data_num}')
    def __getitem__(self, index):
        ##############################################
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform).
        # 3. Return the data (e.g. image and label)
        ##############################################
        img_nr = self.file_[index].split('.')[0]
        #print(img_nr)
        img_file = f'{self.img_p}/{self.phase}/{img_nr}.jpg'
        map_file = f'{self.map_p}/{self.phase}/{img_nr}.png'
        fix_file = f'{self.fix_p}/{self.phase}/{img_nr}.png'
        #print(img_file)
        img_i = cv2.imread(str(img_file))
        img_i = cv2.cvtColor(img_i,cv2.COLOR_BGR2RGB)
        map_i = cv2.imread(str(map_file), cv2.IMREAD_GRAYSCALE)
        fix_i = cv2.imread(str(fix_file), cv2.IMREAD_GRAYSCALE)

        #map_i = np.expand_dims(map_i,axis=-1)
        #fix_i = np.expand_dims(fix_i,axis=-1)

        img_i = self.transform_i(img_i)
        map_i = self.transform_m(map_i)
        fix_i = preprocess_fixmaps(fix_i,self.tar_shape_r,self.tar_shape_c)
        #print(img_i.size())
        return img_i,map_i,fix_i/255.
    def __len__(self):
        ##############################################
        ### Indicate the total size of the dataset
        ##############################################
        return self.data_num

if __name__ == "__main__":
    phase = 'train'
    shape_r,shape_c = 224,320
    batch_size = 16
    #g = generator_SAL(phase,shape_r,shape_c,batch_size)
    g = generator_SAL_torch(phase,shape_r,shape_c)
    train_loader = DataLoader(dataset=g,
                          batch_size=16, 
                          shuffle=True,
                          num_workers=4)

    for img_,map_,fix_ in train_loader:
        print ('Size of image:', img_.size())  # batch_size*3*224*224
        print ('Type of image:', img_.dtype)   # float32
        print ('Size of label:', map_.size())  # batch_size
        print ('Type of label:', map_.dtype)   # int64(long)
        print ('Size of label:', fix_.size())  # batch_size
        print ('Type of label:', fix_.dtype)   # int64(long)
        img_ = img_.numpy()
        map_ = map_.numpy()
        fix_ = fix_.numpy()
        cv2.imshow('w',np.moveaxis(img_[0], [1, 2, 0],[0, 1, 2]))
        cv2.imshow('w1',cv2.cvtColor(np.moveaxis(map_[0], [1, 2, 0],[0, 1, 2]),cv2.COLOR_GRAY2BGR))
        cv2.imshow('w2',cv2.cvtColor(np.moveaxis(fix_[0], [1, 2, 0],[0, 1, 2]),cv2.COLOR_GRAY2BGR))
        cv2.waitKey()
    '''
    for img_,map_,fix_ in g :
        print(img_.shape , map_.shape , fix_.shape,img_.dtype,map_.dtype,fix_.dtype)
        cv2.imshow('w',np.moveaxis(img_[0], [1, 2, 0],[0, 1, 2]))
        cv2.imshow('w1',cv2.cvtColor(np.moveaxis(map_[0], [1, 2, 0],[0, 1, 2]),cv2.COLOR_GRAY2BGR))
        cv2.imshow('w2',cv2.cvtColor(np.moveaxis(fix_[0], [1, 2, 0],[0, 1, 2]),cv2.COLOR_GRAY2BGR))
        cv2.waitKey()
    print('reset')
    for img_,map_,fix_ in g :
        print(img_.shape , map_.shape , fix_.shape)
        cv2.imshow('w',img_[0])
        cv2.imshow('w1',cv2.cvtColor(map_[0],cv2.COLOR_GRAY2BGR))
        cv2.imshow('w2',cv2.cvtColor(fix_[0],cv2.COLOR_GRAY2BGR))
        cv2.waitKey()
    print('done')
    '''
    