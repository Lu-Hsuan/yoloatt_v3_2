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

# using this when you only have a cpu and testing code:
#  python yoloatt_train.py --num_workers 0 --batch_size 4 --no_cuda --epochs 2 --log_period 2 --save_period 1 --debug

class Tester:

    def __init__(self, opt):
        self.opt = opt
        print(self.opt)

        self.device = 'cuda' if not self.opt.no_cuda else 'cpu'
        self.model = Darkmean_stdnet("config/yoloatt.cfg")
        self.model.to(self.device)

        val_dataset = SALCell_Dataset(self.opt.data_path, "val", self.opt.height, self.opt.width)
        self.val_dataloader = DataLoader(val_dataset, self.opt.batch_size, num_workers=self.opt.num_workers,
                                            shuffle=True, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_dataloader)

        self.writer = {}

        self.load_model()

    def val(self):
        self.model.eval()
        with torch.no_grad():
            for inputs,cells,img_nr in self.val_dataloader :
                outputs, losses = self.process(inputs, cells)
                self.save_npy(outputs,img_nr)
                #losses = losses[''].cpu().numpy()
                #np.save(os.path.join(opt.log_path,'val','loss'+'.npy'),losses)
                break


    def process(self, inputs, cells):
        inputs = inputs.to(self.device)

        for i in range(len(cells)):
            cells[i] = cells[i].to(self.device)

        outputs = self.model(inputs)
        losses = self.losses(outputs, cells)
        return outputs, losses

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

    opt = YOLOATT_OPTION().parse()
    path = os.path.join(opt.log_path)
    if os.path.exists(path):
        print(f"{path} already exists")
        exit(-1)
    else:
        os.mkdir(opt.log_path)
        os.mkdir(os.path.join(opt.log_path,'val'))
        os.mkdir(os.path.join(opt.log_path,'val','1'))
        os.mkdir(os.path.join(opt.log_path,'val','2'))
        os.mkdir(os.path.join(opt.log_path,'val','3'))


    yolo_trainer = Tester(opt)
    yolo_trainer.val()