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
from att_dataset import SALCell_Dataset,generator_SAL_metric,remove_padding
from utils import datasets
from models import Darknet
from option import YOLOATT_OPTION
from salience_metrics import *
from model_yoloatt import yolo_att
import cv2
# using this when you only have a cpu and testing code:
#  python yoloatt_train.py --num_workers 0 --batch_size 4 --no_cuda --epochs 2 --log_period 2 --save_period 1 --debug

class Tester:

    def __init__(self, opt):
        self.opt = opt
        print(self.opt)

        self.device = 'cuda' if not self.opt.no_cuda else 'cpu'
        self.model = Darknet(opt.model_cfg)
        self.model.to(self.device)
        self.att_flag = False
        if self.att_flag:
            self.att_model = yolo_att.Darkmean_stdnet("model_yoloatt/yoloatt.cfg")
            self.att_model.to(self.device)
            self.att_model.load_state_dict(torch.load('model_yoloatt/yoloatt_w25.pth'))
        self.tar_shape_r,self.tar_shape_c = 480,640
        self.pad_shape_r,self.pad_shape_c = 416,416
        self.val_dataset = generator_SAL_metric(self.opt.data_path, "val", self.pad_shape_r,self.pad_shape_c,padding=True)
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
                if self.att_flag:
                    att_outs = self.att_model(img_t)
                out = outputs[-1][:,0,:,:]
                out = out.reshape(out.size()[0],1,out.size()[-2],out.size()[-1])
                if self.att_flag:
                    att_out = att_outs[-1][:,0,:,:]
                    att_out = att_out.reshape(att_out.size()[0],1,att_out.size()[-2],att_out.size()[-1])
                    att_logits = nn.functional.interpolate(att_out,size=[self.pad_shape_r,self.pad_shape_c],mode='bilinear')
                #print(out.size())
                logits = nn.functional.interpolate(out,size=[self.pad_shape_r,self.pad_shape_c],mode='bilinear')
                map_p = logits.cpu().numpy().reshape(N,self.tar_shape_r,self.tar_shape_c)
                map_g = map_.numpy().reshape(N,self.tar_shape_r,self.tar_shape_c)
                fix_g = fix_.numpy().reshape(N,self.tar_shape_r,self.tar_shape_c)
                #print(map_p.shape,map_g.shape,fix_g.shape)
                for k in range(N):
                    #map_p_ = cv2.resize(map_p[k], (fix_.shape[-1], fix_.shape[-2]),interpolation=cv2.INTER_LINEAR)
                    #self.Metric_['AUC_J'] += auc_judd(map_p[k],fix_g[k])
                    #self.Metric_['s-AUC'] += auc_shuff_acl(map_p[k],fix_g[k],next(self.o_g))
                    map_pp = remove_padding(map_p[k],[self.tar_shape_r,self.tar_shape_c],[self.pad_shape_r,self.pad_shape_c])
                    self.Metric_['NSS'] += nss(map_pp,fix_g[k])
                    self.Metric_['SIM'] += similarity(map_pp,map_g[k])
                    self.Metric_['CC'] += cc(map_pp,map_g[k])
                    self.Metric_['KL'] += kldiv(map_pp,map_g[k])
                i += 1
                NN += N
                print(i,end='\n')
                if self.att_flag:
                    img_s = img_[-1].cpu().numpy()*255
                    img_s = np.moveaxis(img_s,[0,1,2],[2,0,1])
                    img_s = img_s[:,:,::-1]
                    img_s = cv2.resize(img_s,(self.tar_shape_c,self.tar_shape_r))
                    map_g = map_g[-1][...,np.newaxis]*np.array([255,255,255])
                    map_p = map_pp[...,np.newaxis]*np.array([255,255,255])
                
                    map_a = att_logits.cpu().numpy().reshape(N,self.pad_shape_r,self.pad_shape_c)
                    map_aa = remove_padding(map_a[-1],[self.tar_shape_r,self.tar_shape_c],[self.pad_shape_r,self.pad_shape_c])
                    map_a = map_aa[...,np.newaxis]*np.array([255,255,255])
                #print(img_s.shape , map_g.shape, map_p.shape)
                    img_sa = np.concatenate([img_s,map_g,map_a,map_p],axis=1)
                    cv2.imwrite(f'{os.path.join(opt.log_path,"output_map_pad")}/{img_nr[-1]}.png',np.round(img_sa))
            #print(key)
                if((i+1) % 50 == 0):
                    print(f"eval : {self.opt.weight} , AUC_J: {self.Metric_['AUC_J']/NN:4.4f} , s-AUC: {self.Metric_['s-AUC']/NN:4.4f} , NSS: {self.Metric_['NSS']/NN:4.4f}")
                    print(f"eval : {self.opt.weight} , SIM  : {self.Metric_['SIM']/NN:4.4f} , CC   : {self.Metric_['CC']/NN:4.4f} , KL : {self.Metric_['KL']/NN:4.4f}")
            #if(i == 1):
            #    break
            #print(f'batch:{i:04d},T_loss:{loss.item():4.4f},KL:{loss_p[0].item():4.4f},CC:{loss_p[1].item():4.4f},NSS:{loss_p[2].item():4.4f}')
        for key in self.Metric_:
            #print(key)
            self.Metric_[key] /= self.tot_N
        print(f"eval : {self.opt.weight} , AUC_J: {self.Metric_['AUC_J']:4.4f} , s-AUC: {self.Metric_['s-AUC']:4.4f} , NSS: {self.Metric_['NSS']:4.4f}")
        print(f"eval : {self.opt.weight} , SIM  : {self.Metric_['SIM']:4.4f} , CC   : {self.Metric_['CC']:4.4f} , KL : {self.Metric_['KL']:4.4f}")

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
                self.model.load_darknet_weights("../weights/yolov3_darknet53.conv.74")
            else:
                print("Load Darknet weight")
                self.model.load_darknet_weights("../weights/darknet53.conv.74")
        else:
            print(f"Load Model from {self.opt.weight}")
            if(self.opt.weight.endswith('.weights')):
                self.model.load_darknet_weights(self.opt.weight)
                
            elif(self.opt.weight.endswith('.pth')):
                print('only load model')
                self.model.load_state_dict(torch.load(self.opt.weight))
            else:
                print('load model')
                self.model.load_state_dict(torch.load(self.opt.weight+'/yoloatt_v3_1.pth'))
                #self.model_optimizer.load_state_dict(torch.load(self.opt.weight+'/adam.pth'))

if __name__ == '__main__':
    torch.backends.cudnn.deterministic =True
    opt = YOLOATT_OPTION().parse()
    path = os.path.join(opt.log_path,'output_map')
    os.makedirs(path,exist_ok=True)

    yolo_trainer = Tester(opt)
    yolo_trainer.val()