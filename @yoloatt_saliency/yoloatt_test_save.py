import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter


import os
import numpy as np
from time import time
from att_dataset_test import SALCell_Dataset_test
from yolo_att import Darkmean_stdnet
from option import YOLOATT_OPTION
from PIL import Image
import cv2
# using this when you only have a cpu and testing code:
#  python yoloatt_train.py --num_workers 0 --batch_size 4 --no_cuda --epochs 2 --log_period 2 --save_period 1 --debug

class Tester:

    def __init__(self, opt):
        self.opt = opt
        print(self.opt)

        self.device = 'cuda' if not self.opt.no_cuda else 'cpu'
        self.model = Darkmean_stdnet("config/yoloatt.cfg")
        self.model.to(self.device)

        val_dataset = SALCell_Dataset_test(self.opt.data_path,self.opt.height, self.opt.width)
        self.val_dataloader = DataLoader(val_dataset, self.opt.batch_size, num_workers=self.opt.num_workers,
                                            shuffle=False, pin_memory=True, drop_last=False)
        self.writer = {}

        self.load_model()
        self.tar_shape_r = 0
        self.tar_shape_c = 0
    def val(self):
        self.model.eval()
        with torch.no_grad():
            for inputs,img_nr in self.val_dataloader :
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                #self.save_npy(outputs,img_nr)
                out = outputs[-1][:,0,:,:]
                out = out.reshape(out.size()[0],1,1,out.size()[-2],out.size()[-1])
                #print(out.size())
                for i in range(out.size()[0]):
                    save_p = f"{opt.log_path}/ident/{img_nr[i].split('.')[0]}"
                    os.makedirs(save_p,exist_ok=True)
                    path_ = f'{opt.data_path}/{img_nr[i]}'
                    print(save_p)
                # Create plot
                    img = np.array(Image.open(path_))
                    self.tar_shape_r = img.shape[0]
                    self.tar_shape_c = img.shape[1]
                    logits = nn.functional.interpolate(out[i],size=[self.tar_shape_r,self.tar_shape_c],mode='bilinear')
                    map_p = logits.cpu().numpy().reshape(self.tar_shape_r,self.tar_shape_c)
                    map_ps = np.round(map_p*255)
                    cv2.imwrite(f'{save_p}/map_Pred.png',map_ps)
                    cv2.imwrite(f'{save_p}/img.png',cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                #losses = losses[''].cpu().numpy()
                #np.save(os.path.join(opt.log_path,'val','loss'+'.npy'),losses)

    def save_npy(self,outputs,img_nr):
        for i,(output) in enumerate(outputs):
            output = output.cpu().numpy()
            for n in range(output.shape[0]):
                name = img_nr[n]
                print(output[n].shape)
                print(os.path.join(opt.log_path,'val',str(i+1),name+'.npy'))
                np.save(os.path.join(opt.log_path,'val',str(i+1),name+'.npy'),output[n])

    def load_model(self):
        if self.opt.weight == None:
            print("Model weight is None")
            if self.opt.use_yolov3:
                print("Load YOLOV3_BackBone weight")
                self.model.load_darknet_weights("./weights/yolov3_darknet53.conv.74")
            else:
                print("Load Darknet weight")
                self.model.load_darknet_weights("./weights/darknet53.conv.74")
        else:
            print(f"Load Model from {self.opt.weight}")
            self.model.load_state_dict(torch.load(os.path.join(self.opt.weight, "yoloatt.pth")))

if __name__ == '__main__':

    opt = YOLOATT_OPTION().parse()
    path = os.path.join(opt.log_path)
    if os.path.exists(path):
        print(f"{path} already exists")
        exit(-1)
    else:
        os.mkdir(opt.log_path)
        # os.mkdir(os.path.join(opt.log_path,'val'))
        # os.mkdir(os.path.join(opt.log_path,'val','1'))
        # os.mkdir(os.path.join(opt.log_path,'val','2'))
        # os.mkdir(os.path.join(opt.log_path,'val','3'))

    yolo_trainer = Tester(opt)
    yolo_trainer.val()