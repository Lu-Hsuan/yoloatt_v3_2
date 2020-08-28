import os
import numpy as np
import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class SALCell_Dataset(Dataset):

    def __init__(self, data_path, phase, shape_r, shape_c,flip=False, with_map=False):
        self.phase = phase
        self.shape_r = shape_r
        self.shape_c = shape_c
        self.with_map = with_map
        #self.batch_size = batch_size
        self.img_p = os.path.join(data_path, 'images',self.phase)
        self.map_p = os.path.join(data_path, 'maps',self.phase)
        self.cell_p = os.path.join(data_path,'map_mean' ,self.phase)
        self.file_ = os.listdir(self.img_p)
        self.transform = transforms.Compose([ 
                            transforms.ToPILImage(),
                            transforms.Resize((shape_r, shape_c),interpolation=cv2.INTER_LINEAR),
                            transforms.ToTensor(),
                            #transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            #                    std=[0.229, 0.224, 0.225])
                        ])

        print(f'Dataset : {self.phase}, number : {self.__len__()}')

    def __getitem__(self, index):
        ##############################################
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform).
        # 3. Return the data (e.g. image and label)
        ##############################################
        img_nr = self.file_[index].split('.')[0]
        #print(img_nr)
        img_file = os.path.join(self.img_p, img_nr + '.jpg')
        img_i = cv2.imread(str(img_file))
        img_i = cv2.cvtColor(img_i,cv2.COLOR_BGR2RGB)
        if(flip == True):
            prob = np.random.randint(0,2)
            if(prob == 1):
                img_i = np.flip(img_i,1)
        img_i = self.transform(img_i)

        cells_file = [[], [], []]
        for i in range(3):
            cells_file_temp = np.load(os.path.join(self.cell_p, str(i+1), img_nr + '.npy')) # from low to high resolution
            if(flip == True):
                if(prob == 1):
                    cells_file_temp = np.flip(cells_file_temp,1)
            cells_file[i] = cells_file_temp
            
        if self.with_map:
            map_file = os.path.join(self.map_p, img_nr + '.jpg')
            map_i = cv2.imread(str(map_file))
            map_i = self.transform(map_i)
            cells_file.append(map_i)

        return img_i, cells_file

    def __len__(self):
        ##############################################
        ### Indicate the total size of the dataset
        ##############################################
        return len(self.file_)

class generator_SAL_metric(Dataset):
    
    def __init__(self,data_path, phase, shape_r, shape_c,file_list=None):
        self.phase = phase
        self.shape_r = shape_r
        self.shape_c = shape_c
        #self.batch_size = batch_size
        self.img_p = os.path.join(data_path, 'images',self.phase)
        self.map_p = os.path.join(data_path, 'maps',self.phase)
        self.cell_p = os.path.join(data_path,'map_mean' ,self.phase)
        self.fix_p = os.path.join(data_path, 'fixations_map',self.phase)
        self.count = 0
        if(file_list==None):
            self.file_ = os.listdir(self.img_p)
        else:
            with open(file_list, "r") as file:
                self.file_l = file.readlines()
            self.file_ = [x.rstrip() for x in self.file_l] 
        self.file_.sort(key=lambda x: int(x.split('.')[0].split('_')[2]))
        self.data_num = len(self.file_)
        self.transform_i = transforms.Compose([ 
                        transforms.ToPILImage(),
                        transforms.Resize((shape_r, shape_c),interpolation=cv2.INTER_LINEAR),
                        transforms.ToTensor(),
                        #transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        #                    std=[0.229, 0.224, 0.225])
                    ])
        """
        self.transform_m = transforms.Compose([ 
                        transforms.ToPILImage(),
                        transforms.Resize((tar_shape_r, tar_shape_c),interpolation=cv2.INTER_LINEAR),
                        transforms.ToTensor()
                    ])
        """
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
        img_file = f'{self.img_p}/{img_nr}.jpg'
        map_file = f'{self.map_p}/{img_nr}.png'
        fix_file = f'{self.fix_p}/{img_nr}.png'
        #print(img_file)
        img_i = cv2.imread(str(img_file))
        img_i = cv2.cvtColor(img_i,cv2.COLOR_BGR2RGB)
        map_i = cv2.imread(str(map_file), cv2.IMREAD_GRAYSCALE)
        fix_i = cv2.imread(str(fix_file), cv2.IMREAD_GRAYSCALE)

        #map_i = np.expand_dims(map_i,axis=-1)
        #fix_i = np.expand_dims(fix_i,axis=-1)

        img_i = self.transform_i(img_i)
        #map_i = self.transform_m(map_i)
        #fix_i = preprocess_fixmaps(fix_i,self.tar_shape_r,self.tar_shape_c)
        #print(img_i.size())
        return img_i,map_i/255.,fix_i/255.,img_nr
    def __len__(self):
        ##############################################
        ### Indicate the total size of the dataset
        ##############################################
        return self.data_num