import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter


import os
import numpy
from time import time
from att_dataset import SALCell_Dataset
from yolo_att import Darkmean_stdnet
from option import YOLOATT_OPTION

# using this when you only have a cpu and testing code:
#  python yoloatt_train.py --num_workers 0 --batch_size 4 --no_cuda --epochs 2 --log_period 2 --save_period 1 --debug

class Trainer:

    def __init__(self, opt):
        self.opt = opt
        print(self.opt)

        self.device = 'cuda' if not self.opt.no_cuda else 'cpu'
        self.model = Darkmean_stdnet("config/yoloatt.cfg")
        self.model.to(self.device)
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), self.opt.lr)
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

    def train(self):
        self.step = 0
        self.epoch = 0
        for self.epoch in range(1, self.opt.epochs + 1):
            self.run_epoch()

            if self.epoch % self.opt.save_period == 0:
                self.save_model(self.epoch)

        self.save_model(self.epoch)
        print("Finish")


    def run_epoch(self):
        self.model.train()

        log_losses= {'0': 0., '1': 0., '2': 0., 'total': 0.}
        for i, (inputs, cells) in enumerate(self.train_dataloader):
            if self.opt.debug: 
                print("At run_epoch, input images size: ", inputs.size(), "input cells size", [cell.size() for cell in cells])

            start_time = time()
            outputs, losses = self.process(inputs, cells)

            if self.opt.debug: 
                print("At run_epoch, outputs size: ", [output.size() for output in outputs], "losses value:", [(k, round(v.item(), 3)) for k, v in losses.items()])

            for k in log_losses.keys():
                log_losses[k] += losses[k]

            self.model_optimizer.zero_grad()
            losses['total'].backward()
            self.model_optimizer.step()
            
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

                print('Epoch: {} / Batch: {}, Loss: {:.3f} (process time: {:.3f}, left time: {})'
                        .format(self.epoch, i, log_losses['total'], duration, "{:02d}h{:02d}m{:02d}s".format(h, m, s)))

                for k in log_losses.keys():
                    log_losses[k] = 0.

                self.val()

            self.step += 1
        self.model_lr_scheduler.step()


    def val(self):
        self.model.eval()
        try:
            inputs, cells = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs, cells = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process(inputs, cells)
            self.log('val', inputs, cells, outputs, losses)

        self.model.train()


    def process(self, inputs, cells):
        inputs = inputs.to(self.device)

        for i in range(len(cells)):
            cells[i] = cells[i].to(self.device)

        outputs = self.model(inputs)
        losses = self.losses(outputs, cells)
        return outputs, losses
        

    def losses(self, preds, targets):
        losses = {'total': 0.}
        
        for i, (pred, target) in enumerate(zip(preds, targets)):
            losses[f'{i}'] = nn.BCELoss()(pred, target)
            losses['total'] += losses[f'{i}'] / (2 ** (2 - i))   # preds size: 1/32, 1/16, 1/8 => the smaller resolution get smaller weight

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

        save_path = os.path.join(path, "yoloatt.pth")
        torch.save(self.model.state_dict(), save_path)

        save_path = os.path.join(path, "adam.pth")
        torch.save(self.model_optimizer.state_dict(), save_path)


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
            self.model_optimizer.load_state_dict(torch.load(os.path.join(self.opt.weight, "adam.pth")))

if __name__ == '__main__':

    opt = YOLOATT_OPTION().parse()

    path = os.path.join(opt.log_path, opt.model_name)
    if os.path.exists(path):
        print(f"{path} already exists")
        exit(-1)

    yolo_trainer = Trainer(opt)
    yolo_trainer.train()