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
from att_dataset import SALCell_Dataset
from utils import datasets
import models 
import yolo_model 

device = 'cpu'
model_def = 'yolov3.cfg'
weights_path = 'yolov3.weights'
model = yolo_model.Darknet(model_def, img_size=416).to(device)
model.load_darknet_weights(weights_path)
#torch.save(model.state_dict(),'yolov3_w.pth')

model_def_ = 'yoloatt_v3.cfg'
weights_path_ = 'yolov3_w.pth'
model_ = models.Darknet(model_def_, img_size=416).to(device)

for pth, model_a in [[weights_path_, model_]]:
    prepared_dict = torch.load(pth)
    model_dict = model_a.state_dict()
    for k, v in prepared_dict.items():
        if k in model_dict:
            if v.size() == model_dict[k].size():
                model_dict[k]=v
                #print('load')
            else:
                model_dict[k][:-2] = v
                #print('output_c')
        else:
            continue
    #'''
    model_a.load_state_dict(model_dict)
torch.save(model_.state_dict(),'yoloatt_v3_w.pth')
print('calc')
d=torch.rand(2,3,416,416)
targets = torch.rand((4,6))
_,y = model(d,targets=targets)
__,yy = model_(d,targets=targets)
print(_,y[0,0:1,0:10])
print(__,yy[0,0:1,0:10])

