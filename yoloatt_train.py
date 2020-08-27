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
from models import Darknet
from option import YOLOATT_OPTION
import warnings
#warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=Warning)
# using this when you only have a cpu and testing code:
#  python yoloatt_train.py --num_workers 0 --batch_size 4 --no_cuda --epochs 2 --log_period 2 --save_period 1 --debug

class Trainer:

    def __init__(self, opt):
        self.opt = opt
        print(self.opt)

        self.device = 'cuda' if not self.opt.no_cuda else 'cpu'
        self.model = Darknet("./yoloatt_v3.cfg")
        self.model.to(self.device)
        #self.model_optimizer = torch.optim.Adam(self.model.parameters(), self.opt.lr)
        #'''
        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in dict(self.model.named_parameters()).items():
            if '.bias' in k:
                pg2 += [v]  # biases
            elif 'conv' in k:
                pg1 += [v]  # apply weight_decay
            else:
                pg0 += [v]  # all else
        self.model_optimizer = torch.optim.SGD(self.model.parameters(), lr=opt.lr, momentum=0.937, nesterov=True)
        #self.model_optimizer.add_param_group({'params': pg1}) #, 'weight_decay': 0.5})  # add pg1 with weight_decay
        #self.model_optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
        print('Optimizer groups: %g .bias, %g Conv2d.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
        #del pg0, pg1, pg2
        #'''
        self.model_lr_scheduler = torch.optim.lr_scheduler.StepLR(
                                    self.model_optimizer, self.opt.decay_step, self.opt.decay_factor)


        train_dataset = SALCell_Dataset(self.opt.data_path, "train", self.opt.height, self.opt.width)
        val_dataset = SALCell_Dataset(self.opt.data_path, "val", self.opt.height, self.opt.width)

        
        self.train_dataloader = DataLoader(train_dataset, self.opt.batch_size, num_workers=self.opt.num_workers,
                                            shuffle=True, pin_memory=True, drop_last=True)
        self.val_dataloader = DataLoader(val_dataset, self.opt.batch_size, num_workers=self.opt.num_workers,
                                            shuffle=True, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_dataloader)

        self.total_train = self.train_dataloader.__len__() * self.opt.epochs

        self.writer = {}
        for mode in ['train', 'val']:
            self.writer[mode] = SummaryWriter(os.path.join(self.opt.log_path, self.opt.model_name, mode))

        self.load_model()

        if self.opt.see_grad:
            self.mode = 1

        ###################################################################
        class_names = load_classes(self.opt.obj_names)

        obj_dataset = ListDataset('../data/coco/trainvalno5k.txt', augment=True, multiscale=self.opt.multiscale_training)
        self.obj_dataloader = torch.utils.data.DataLoader(
            obj_dataset,
            batch_size=self.opt.batch_size,
            shuffle=True,
            num_workers=self.opt.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=obj_dataset.collate_fn
        )

        self.obj_iter = iter(self.obj_dataloader)

        val_obj_dataset = ListDataset('../data/coco/5k.txt', augment=True, multiscale=self.opt.multiscale_training)
        self.val_obj_dataloader = torch.utils.data.DataLoader(
            val_obj_dataset,
            batch_size=self.opt.batch_size,
            shuffle=False,
            num_workers=self.opt.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=val_obj_dataset.collate_fn
        )

        self.val_obj_iter = iter(self.val_obj_dataloader)
        ###################################################################
        print(obj_dataset.__len__(),val_obj_dataset.__len__())
        self.obj_flag = False
    def train(self):
        self.step = 0
        self.epoch = 0
        for self.epoch in range(1, self.opt.epochs + 1):
            self.run_epoch()

            if self.epoch % self.opt.save_period == 0:
                self.save_model(self.epoch)

        self.save_model(self.epoch)
        print("Finish")


    def see_grad(self, net, mode):
        i, s = 0,  []
        for n, p in net.named_parameters():
            if p.grad is not None:
                i += 1
                s.append(torch.abs(p.grad).mean())
                if mode > 0:
                    print("attention grad: ", n, s[-1])
                else:
                    print("object grad: ", n, s[-1])
        print(f"max: {max(s)}, mean: {sum(s) / i}, min: {min(s)}")
        input()
        print('Change')


    def run_epoch(self):
        self.model.train()

        log_losses= {'0': 0., '1': 0., '2': 0., 'total': 0., 'obj_loss': 0.}
        for i, (inputs, cells) in enumerate(self.train_dataloader):
            if self.opt.debug: 
                print("At run_epoch, input images size: ", inputs.size(), "input cells size", [cell.size() for cell in cells])

            start_time = time()
                
            outputs, losses = self.process(inputs, cells)
            if(self.obj_flag==True):
                ##########################################################
                try:
                    _, imgs, targets = self.obj_iter.next()
                except StopIteration:
                    print('Re')
                    self.obj_iter = iter(self.obj_dataloader)
                    _, imgs, targets = self.obj_iter.next()

                imgs = imgs.to(self.device)
                targets = targets.to(self.device)
                losses['obj_loss'], _, _ = self.model(imgs, targets)
                obj_loss = losses['obj_loss']
                ##########################################################
                #print(losses['obj_loss'])
                total_loss = losses['total'] + self.opt.loss_weight * losses['obj_loss']
            else:
                losses['obj_loss'] = 0
                total_loss = losses['total']
                if(float(losses['total'].cpu().detach().numpy()) < 0.2):
                    self.obj_flag==True
            #total_loss = losses['obj_loss']
            ########################################
            if not self.opt.see_grad:
                self.model_optimizer.zero_grad()
                total_loss.backward()
                self.model_optimizer.step()
            else:
                self.model_optimizer.zero_grad()
                if self.mode > 0:
                    losses['total'].backward()
                else:
                    losses['obj_loss'].backward()
                self.see_grad(self.model, self.mode)
                self.mode *= -1 
                self.model_optimizer.step()
                continue
            #######################################


            if self.opt.debug: 
                print("At run_epoch, outputs size: ", [output.size() for output in outputs], "losses value:", [(k, round(v.item(), 3)) for k, v in losses.items()])

            for k in log_losses.keys():
                log_losses[k] += losses[k]
            
            duration = time() - start_time

            if self.opt.debug:
                print("Batch: {}, Loss: {:.3f}, process time: {:.3f}".format(i, losses['total'], duration))

            if i % self.opt.log_period == 0 and i != 0:
                for k in log_losses.keys():
                    log_losses[k] /= self.opt.log_period
                self.log('train', inputs, cells, outputs, log_losses)

                left_time = int((self.total_train - self.step) * duration)
                s = left_time % 60
                left_time //= 60
                m = left_time % 60
                h = left_time // 60

                print('Epoch: {} / Batch: {}, Att Loss: {:.3f}, Obj Loss: {:.3f} (process time: {:.3f}, left time: {})'
                        .format(self.epoch, i, log_losses['total'], log_losses['obj_loss'], duration, "{:02d}h{:02d}m{:02d}s".format(h, m, s)))

                for k in log_losses.keys():
                    log_losses[k] = 0.

                self.val()
            #if(i==100):
            #    break
            self.step += 1
        self.model_lr_scheduler.step()


    def val(self):
        self.model.eval()
        loss_mean = {'0': 0., '1': 0., '2': 0., 'total': 0., 'obj_loss': 0.}
        t = 10
        for i in range(0,t):
            try:
                inputs, cells = self.val_iter.next()
            except StopIteration:
                self.val_iter = iter(self.val_dataloader)
                inputs, cells = self.val_iter.next()

            ################################################################
            try:
                _, imgs, targets = self.val_obj_iter.next()
            except StopIteration:
                self.val_obj_iter = iter(self.val_obj_dataloader)
                _, imgs, targets = self.val_obj_iter.next()

            imgs = imgs.to(self.device)
            targets = targets.to(self.device)

            with torch.no_grad():
                outputs, losses = self.process(inputs, cells)
                losses['obj_loss'], _, _ = self.model(imgs, targets)
                for k in loss_mean.keys():
                    loss_mean[k] += losses[k]
            
        for k in loss_mean.keys():
            loss_mean[k] = loss_mean[k]/t
        print(f"val loss: attention: {loss_mean['total']}, obj: {loss_mean['obj_loss']}")
            #################################################################

        self.log('val', inputs, cells, outputs, loss_mean)
        self.model.train()


    def process(self, inputs, cells):
        inputs = inputs.to(self.device)

        for i in range(len(cells)):
            cells[i] = cells[i].to(self.device)

        _, _, outputs = self.model(inputs)
        losses = self.losses(outputs, cells)
        return outputs, losses
        

    def losses(self, preds, targets):
        losses = {'total': 0.}
        for i, (pred, target) in enumerate(zip(preds, targets)):
            losses[f'{i}'] = nn.BCELoss()(pred, target)
            losses['total'] += losses[f'{i}']# / (2 ** (2 - i))   # preds size: 1/32, 1/16, 1/8 => the smaller resolution get smaller weight

        losses['total'] /= (len(losses.keys()) - 1)
        return losses


    def log(self, mode, inputs, cells, outputs, losses):
        inputs = inputs.to(self.device)
        for i in range(len(cells)):
            cells[i] = cells[i].to(self.device)
            
        for k in losses.keys():
            self.writer[mode].add_scalar(f'loss_{k}', losses[k], self.step)

        images = []
        for cell, output in zip(cells, outputs):
            if self.opt.debug:
                print(f"At log {mode}: inputs[0] size: ", inputs[0].size(), 
                    "cell[0, 0:1] size: ", cell[0, 0:1].expand(3, -1, -1).size(),
                     "output[0, 0:1] size: ", output[0, 0:1].expand(3, -1, -1).size())


            cell = F.interpolate(cell, size=(self.opt.height, self.opt.width))
            output = F.interpolate(output, size=(self.opt.height, self.opt.width))

            images.append(torch.cat([inputs[0], cell[0, 0:1].expand(3, -1, -1), output[0, 0:1].expand(3, -1, -1)], axis=-1))
            images.append(torch.cat([inputs[0], cell[0, 1:].expand(3, -1, -1), output[0, 1:].expand(3, -1, -1)], axis=-1))

        image = torch.cat(images, axis=-2)

        self.writer[mode].add_image('image', image, self.step)


    def save_model(self, i):
        path = os.path.join(self.opt.log_path, self.opt.model_name, 'model', f'weight_{i}')
        if not os.path.exists(path):
            os.makedirs(path)
        
        print(f"saving {path}")

        save_path = os.path.join(path, "yoloatt_v3.pth")
        torch.save(self.model.state_dict(), save_path)

        save_path = os.path.join(path, "sgd.pth")
        torch.save(self.model_optimizer.state_dict(), save_path)


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
            if(self.opt.weight.endswith('.pth')):
                #################################################################################################################
                for pth, model in [[self.opt.weight, self.model]]:
                    prepared_dict = torch.load(pth)
                    model_dict = model.state_dict()
                    for k, v in prepared_dict.items():
                        if k in model_dict:
                            if v.size() == model_dict[k].size():
                                model_dict[k]=v
                                #print('load')
                            else:
                                if("yoloatt.pth" in self.opt.weight):
                                    model_dict[k][-2:] = v
                                else:
                                    model_dict[k][:-2] = v
                                #print('output_c')
                        else:
                            continue
                    #'''
                    model.load_state_dict(model_dict)
                    break
            else:
                print('load model and opti')
                self.model.load_state_dict(torch.load(self.opt.weight+'/yoloatt_v3.pth'))
                self.model_optimizer.load_state_dict(torch.load(self.opt.weight+'/sgd.pth'))
            #################################################################################################################
if __name__ == '__main__':

    opt = YOLOATT_OPTION().parse()

    path = os.path.join(opt.log_path, opt.model_name)
    if os.path.exists(path):
        print(f"{path} already exists")
        key = input("Continue [Y/N] ?")
        if key == 'N': exit(-1)
    else:
        os.makedirs(path)

    to_save = opt.__dict__.copy()
    with open(os.path.join(path, 'opt.json'), 'w') as f:
        json.dump(to_save, f, indent=2)

    yolo_trainer = Trainer(opt)
    yolo_trainer.train()