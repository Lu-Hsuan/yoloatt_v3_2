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
#from att_dataset import SALCell_Dataset
from utils import datasets
import models 
import models_2

device = 'cpu'
'''
model_def = 'yoloatt_v3_2.cfg'
weights_path = 'yoloatt_v3_2.pth'
model = models_2.Darknet(model_def, img_size=416).to(device)
model.load_state_dict(torch.load(weights_path))
#torch.save(model.state_dict(),'yolov3_w.pth')
'''
model_def_ = 'yoloatt_v3_1.cfg'
weights_path_ = 'yoloatt_v3_2.pth' #model2 weight (epoch5)
model_ = models.Darknet(model_def_, img_size=416).to(device)

for pth, model_a in [[weights_path_, model_]]:
    prepared_dict = torch.load(pth)
    model_dict = model_a.state_dict()
    for k, v in prepared_dict.items():
        if k in model_dict:
            if v.size() == model_dict[k].size():
                model_dict[k] = v
                print('load',k)
        else:
            continue
model_a.load_state_dict(model_dict)
torch.save(model_.state_dict(),'yoloatt_v3_1.pth')

