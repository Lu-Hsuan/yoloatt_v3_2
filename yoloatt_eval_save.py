import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import numpy
import json
from time import time
from att_dataset import SALCell_Dataset,generator_SAL_metric
from utils import datasets
from models import Darknet
from option import YOLOATT_OPTION
from salience_metrics import *
import cv2
# using this when you only have a cpu and testing code:
#  python yoloatt_train.py --num_workers 0 --batch_size 4 --no_cuda --epochs 2 --log_period 2 --save_period 1 --debug

class Tester:

    def __init__(self, opt):
        self.opt = opt
        print(self.opt)

        self.device = 'cuda' if not self.opt.no_cuda else 'cpu'
        self.model = Darknet("./yoloatt_v3.cfg")
        self.model.to(self.device)
        self.tar_shape_r,self.tar_shape_c = 480,640

        self.val_dataset = generator_SAL_metric(self.opt.data_path, "val", self.opt.height, self.opt.width,file_list='./data/common_sal.txt')
        self.val_dataloader = DataLoader(self.val_dataset, self.opt.batch_size, num_workers=self.opt.num_workers,
                                            shuffle=False, pin_memory=True, drop_last=False)

        self.tot_N = self.val_dataset.data_num
        #self.g = generator_SAL('val',self.opt.height,self.opt.width,self.opt.batch_size, data_path='./val_data')
        #self.o_g = other_maps(self.g,10,self.tar_shape_r,self.tar_shape_c)
        self.Metric_ = {'AUC_J':0.0,'SIM':0.0,'s-AUC':0.0,'CC':0.0,'NSS':0.0,'KL':0.0}
        self.writer  = {}

        self.load_model()

    def val(self):
        i = 0
        NN = 0
        self.model.eval()
        with torch.no_grad():
            for img_,map_,fix_,img_nr in self.val_dataloader :
                N = img_.size()[0]
                img_t = img_.to(self.device)
                _,_,outputs = self.model(img_t)
                self.save_npy(outputs,img_nr)
                #print(map_p.shape,map_g.shape,fix_g.shape)
                i += 1
                NN += N
                print(i,end='\n')
            print(NN)
            #print(key)
            #if(i == 1):
            #    break
            #print(f'batch:{i:04d},T_loss:{loss.item():4.4f},KL:{loss_p[0].item():4.4f},CC:{loss_p[1].item():4.4f},NSS:{loss_p[2].item():4.4f}')
    def save_npy(self,outputs,img_nr):
        for i,(output) in enumerate(outputs):
            output = output.cpu().numpy()
            for n in range(output.shape[0]):
                name = img_nr[n]
                #print(output[n].shape)
                #print(os.path.join(opt.log_path,'val',str(i+1),name+'.npy'))
                np.save(os.path.join(opt.log_path,'val',str(i+1),name+'.npy'),output[n])

    def load_model(self):
        if self.opt.weight == None:
            print("Model weight is None")
            if self.opt.use_yolov3:
                print("Load YOLOV3_BackBone weight")
                self.model.load_darknet_weights("../weights/yolov3_darknet53.conv.74")
            else:
                print("Load Darknet weight")
                self.model.load_darknet_weights("../weights/darknet53.conv.74")
        else:
            print(f"Load Model from {self.opt.weight}")
            if(self.opt.weight.endswith('.weights')):
                self.model.load_darknet_weights(self.opt.weight)
                
            if('yoloatt_v3.pth' in self.opt.weight):
                print('only load model')
                self.model.load_state_dict(torch.load(self.opt.weight))
            else:
                print('load model')
                self.model.load_state_dict(torch.load(self.opt.weight+'/yoloatt_v3.pth'))
                #self.model_optimizer.load_state_dict(torch.load(self.opt.weight+'/sgd.pth'))

if __name__ == '__main__':
    torch.backends.cudnn.deterministic =True
    opt = YOLOATT_OPTION().parse()
    if os.path.exists(opt.log_path):
        opt.log_path = opt.log_path+'_2'
        print(f'exist ch to{opt.log_path}')
    os.makedirs(os.path.join(opt.log_path,'val','1'))
    os.makedirs(os.path.join(opt.log_path,'val','2'))
    os.makedirs(os.path.join(opt.log_path,'val','3'))

    yolo_trainer = Tester(opt)
    yolo_trainer.val()