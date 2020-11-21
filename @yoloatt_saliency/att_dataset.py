import os
import numpy as np
import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class SALCell_Dataset(Dataset):

    def __init__(self, data_path, phase, shape_r, shape_c):
        self.phase = phase
        self.shape_r = shape_r
        self.shape_c = shape_c
        #self.batch_size = batch_size
        self.img_p = os.path.join(data_path, 'images',self.phase)
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
        img_i = self.transform(img_i)
        
        cells_file = [[], [], []]
        for i in range(3):
            cells_file[i] = np.load(os.path.join(self.cell_p, str(i+1), img_nr + '.npy'))
            
        return img_i, cells_file

    def __len__(self):
        ##############################################
        ### Indicate the total size of the dataset
        ##############################################
        return len(self.file_)