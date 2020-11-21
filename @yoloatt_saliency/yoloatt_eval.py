import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import os
import numpy as np
from time import time
from att_dataset_test import SALCell_Dataset
from yolo_att import Darkmean_stdnet
from option import YOLOATT_OPTION
from data_gen import *
from salience_metrics import *
# using this when you only have a cpu and testing code:
#  python yoloatt_train.py --num_workers 0 --batch_size 4 --no_cuda --epochs 2 --log_period 2 --save_period 1 --debug

class Tester:

    def __init__(self, opt):
        self.opt = opt
        print(self.opt)

        self.device = 'cuda' if not self.opt.no_cuda else 'cpu'
        self.model = Darkmean_stdnet("config/yoloatt.cfg")
        self.model.to(self.device)
        self.tar_shape_r,self.tar_shape_c = 480,640

        self.val_g = generator_SAL_metric('val',self.opt.height,self.opt.width,tar_shape_r=self.tar_shape_r,tar_shape_c=self.tar_shape_c)
        self.val_loader = DataLoader(dataset=self.val_g,
                                batch_size=self.opt.batch_size, 
                                shuffle=False,
                                num_workers=self.opt.num_workers)

        self.tot_N = self.val_g.data_num
        self.g = generator_SAL('val',self.opt.height,self.opt.width,self.opt.batch_size,data_path=self.opt.data_path)
        self.o_g = other_maps(self.g,10,self.tar_shape_r,self.tar_shape_c)
        self.Metric_ = {'AUC_J':0.0,'SIM':0.0,'s-AUC':0.0,'CC':0.0,'NSS':0.0,'KL':0.0}
        self.writer = {}

        self.load_model()

    def val(self):
        i = 0
        NN = 0
        self.model.eval()
        with torch.no_grad():
            for img_,map_,fix_ in self.val_loader :
                N = img_.size()[0]
                img_t = img_.to(self.device)
                outputs = self.model(img_t)
                out = outputs[-1][:,0,:,:]
                out = out.reshape(out.size()[0],1,out.size()[-2],out.size()[-1])
                print(out.size())
                logits = nn.functional.interpolate(out,size=[self.tar_shape_r,self.tar_shape_c],mode='bilinear')
                map_p = logits.cpu().numpy().reshape(N,self.tar_shape_r,self.tar_shape_c)
                map_g = map_.numpy().reshape(N,self.tar_shape_r,self.tar_shape_c)
                fix_g = fix_.numpy().reshape(N,self.tar_shape_r,self.tar_shape_c)
                #print(map_p.shape,map_g.shape,fix_g.shape)
                for k in range(N):
                    #map_p_ = cv2.resize(map_p[k], (fix_.shape[-1], fix_.shape[-2]),interpolation=cv2.INTER_LINEAR)
                    self.Metric_['AUC_J'] += auc_judd(map_p[k],fix_g[k])
                    self.Metric_['s-AUC'] += auc_shuff_acl(map_p[k],fix_g[k],next(self.o_g))
                    self.Metric_['NSS'] += nss(map_p[k],fix_g[k])
                    self.Metric_['SIM'] += similarity(map_p[k],map_g[k])
                    self.Metric_['CC'] += cc(map_p[k],map_g[k])
                    self.Metric_['KL'] += kldiv(map_p[k],map_g[k])
                i += 1
                NN += N
                print(i,end='\n')
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


    def losses(self, preds, targets):
        losses = {'total': 0.}
        for i, (pred, target) in enumerate(zip(preds, targets)):
            losses[f'{i}'] = nn.BCELoss()(pred, target)
            losses['total'] += losses[f'{i}'] / (2 ** (2 - i))   # preds size: 1/32, 1/16, 1/8 => the smaller resolution get smaller weight

        return losses

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
    torch.backends.cudnn.deterministic =True
    opt = YOLOATT_OPTION().parse()
    path = os.path.join(opt.log_path)

    yolo_trainer = Tester(opt)
    yolo_trainer.val()