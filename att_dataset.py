import os
import numpy as np
import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
    def __call__(self, tensor):
        t = tensor + torch.randn(tensor.size()) * self.std + self.mean
        t = torch.clamp(t, 0, 1)
        return t
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class SALCell_Dataset(Dataset):
    def __init__(self, data_path, phase, shape_r, shape_c,augment=False, with_map=False):
        self.phase = phase
        self.shape_r = shape_r
        self.shape_c = shape_c
        self.with_map = with_map
        #self.batch_size = batch_size
        self.img_p = os.path.join(data_path, 'images',self.phase)
        self.map_p = os.path.join(data_path, 'maps',self.phase)
        self.cell_p = os.path.join(data_path,'map_mean' ,self.phase)
        self.file_ = os.listdir(self.img_p)
        self.file_.sort(key=lambda x: int(x.split('.')[0].split('_')[2]))
        if(augment):
            self.transform = transforms.Compose([ 
                            transforms.ToPILImage(),
                            transforms.Resize((shape_r, shape_c),interpolation=cv2.INTER_LINEAR),
                            transforms.ColorJitter(hue=(-0.1, 0.1), saturation=(0.8, 1.2), brightness=(0.8, 1.2), contrast=(0.8, 1.2)),
                            transforms.ToTensor()
                            #transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            #                    std=[0.229, 0.224, 0.225])
                        ])
        else:
            self.transform = transforms.Compose([ 
                            transforms.ToPILImage(),
                            transforms.Resize((shape_r, shape_c),interpolation=cv2.INTER_LINEAR),
                            #transforms.ColorJitter(hue=(-0.1, 0.1), saturation=(0.8, 1.2), brightness=(0.8, 1.2), contrast=(0.8, 1.2)),
                            transforms.ToTensor()
                            #transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            #                    std=[0.229, 0.224, 0.225])
                        ])
        print(f'Dataset : {self.phase}, number : {self.__len__()}')
        self.augment = augment
        self.GaussianNoise = AddGaussianNoise(0,0.02)
    def __getitem__(self, index):
        ##############################################
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform).
        # 3. Return the data (e.g. image and label)
        ##############################################
        img_nr = self.file_[index].split('.')[0]
        
        img_file = os.path.join(self.img_p, img_nr + '.jpg')
        #print(img_file)
        img_i = cv2.imread(str(img_file))
        img_i = cv2.cvtColor(img_i,cv2.COLOR_BGR2RGB)
        if(self.augment == True):
            prob_f = np.random.randint(0,2)
            #print(prob_f)
            if(prob_f == 1):
                img_i = np.flip(img_i,1)
        img_i = self.transform(img_i)

        cells_file = [[], [], []]
        for i in range(3):
            cells_file_temp = np.load(os.path.join(self.cell_p, str(i+1), img_nr + '.npy')) # from low to high resolution
            if(self.augment == True):
                if(prob_f == 1):
                    cells_file_temp = np.flip(cells_file_temp,2)
                    #print(cells_file_temp.shape)
            cells_file[i] = cells_file_temp.copy()

        if self.with_map:
            map_file = os.path.join(self.map_p, img_nr + '.jpg')
            map_i = cv2.imread(str(map_file))
            map_i = self.transform(map_i)
            cells_file.append(map_i)

        if(self.augment == True):
            prob_n = np.random.randint(0,2)
            #print(prob_n)
            if(prob_n == 1):
                img_i = self.GaussianNoise(img_i)
        print(img_nr,'noise',prob_n,'flip',prob_f)
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
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator
    #'''
    train_dataset = SALCell_Dataset('D:/!@c++/D_DP_model/train_1', "train", 224, 320,augment=True)
    #train_dataset = SALCell_Dataset('../data', "train", 224, 320,augment=True)
    train_dataloader = DataLoader(train_dataset,1, num_workers=2,
                                        shuffle=False, pin_memory=False, drop_last=True)
    img_p = r'D:\!@c++\D_DP_model\train_1\images\train'.replace('\\'[0],'/')
    img_file_ = os.listdir(img_p)
    i=0                    
    for img_i,cell in train_dataloader:
        img_i = img_i[0].numpy()
        img_i = np.moveaxis(img_i,[0,1,2],[2,0,1])
        cell = cell[-1][0].numpy()
        print(img_i.shape,cell.shape)
        fig,ax = plt.subplots(ncols=2,figsize=(8,4),dpi=100)
        ax[0].imshow(img_i)
        ax[1].imshow(cell[0])
        ax[0].axis("off")
        ax[1].axis("off")
        fig.savefig(f'!img/Dataset_{img_file_[i]}')
        plt.show()
        print()
        i+=1
    #'''
    
    """
    img_p = r'D:\!@c++\D_DP_model\train_1\images\train'.replace('\\'[0],'/')
    img_file_ = os.listdir(img_p)[0:10]
    linear = False
    save = True
    for img_file in img_file_:
        print(img_file)
        img_ = cv2.imread(f'{img_p}/{img_file}')
        img_ = cv2.cvtColor(img_,cv2.COLOR_BGR2RGB)
        GaussianNoise = AddGaussianNoise(0,0.02)
        fig,ax = plt.subplots(ncols=5,figsize=(20,16),dpi=100)
        for i in range(5):
            if(linear):
                delta = i/10
                h_b = 0.2
                base = 0.8
                H = delta-h_b
                SBC = base+delta
                range_ = 0
                print(delta)
            else:
                H = 0
                SBC = 1
                range_SBC = 0.2
                range_H = 0.1
                H_L = H-range_H
                H_H = H+range_H
                SBC_L = SBC-range_SBC
                SBC_H = SBC+range_SBC
            
            transform_ = transforms.Compose([ 
                        transforms.ToPILImage(),
                        transforms.Resize((224, 320),interpolation=cv2.INTER_LINEAR),
                        transforms.ColorJitter(hue=(H_L,H_H), saturation=(SBC_L, SBC_H), \
                        brightness=(SBC_L, SBC_H), contrast=(SBC_L, SBC_H)),
                        #transforms.ColorJitter(hue=(0+delta-h_b,0+delta-h_b), saturation=(base, base),\
                        #    brightness=(base+0, base+0), contrast=(base+0, base+0)),
                        transforms.ToTensor()
                        #transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        #                    std=[0.229, 0.224, 0.225])
                    ])
            img_i = transform_(img_.copy())
            img_i = GaussianNoise(img_i)
            img_i = img_i.numpy()
            img_i = np.moveaxis(img_i,[0,1,2],[2,0,1])
            ax[i].imshow(img_i)
            #ax[i].set_title(f'H:{delta-h_b:.2f},S,B,C:{base:.2f}')
            if(linear):
                ax[i].set_title(f'H:{H:.2f},S,B,C:{SBC:.2f}')
            else:
                ax[i].set_title(f'H:{H_L:.2f}~{H_H:.2f},S,B,C:{SBC_L:.2f}~{SBC_H:.2f}')
            ax[i].axis("off")
            
        fig.set_size_inches(1600/100.0, 800/100.0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
        plt.margins(0,0)
        if(save):
            fig.savefig(f'!img/Range_{img_file}')
            plt.close()
        else:
            plt.show()
    """