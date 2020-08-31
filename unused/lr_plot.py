from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

#from utils.parse_config import *
#from utils.utils import build_targets, to_cpu, non_max_suppression

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

if __name__ == "__main__":
    step_time = 936
    d=torch.rand(2,3,4,4)
    model_optimizer = torch.optim.Adam([d], 7e-4)
    #print(y[-1].size())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optimizer, T_max=936,eta_min=1e-6,last_epoch=-1)
    y = []
    epochs = 312*3*5
    for _ in range(epochs):
        if((_) % 936 == 0 and _ != 0):
            model_optimizer.param_groups[0]['initial_lr'] = 7e-4*0.7**(_/936+1)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optimizer, T_max=936,eta_min=1e-6,last_epoch=-1)
            #model_optimizer.defaults['lr'] = 1e-4
        
        scheduler.step()
        y.append(model_optimizer.param_groups[0]['lr'])
        
    plt.plot(y, '.-', label='CosineLR')
    plt.xlabel('step')
    plt.ylabel('LR')
    plt.tight_layout()
    plt.show()
