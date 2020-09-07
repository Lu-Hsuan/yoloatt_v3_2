from __future__ import division

from models_rect import *
from utils.utils import *
from utils.datasets import *
from att_dataset import generator_SAL_test

import os
import sys
import time
import datetime
import argparse
import tqdm

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

import cv2

def find_max(mean_,std_,obj_):
    inx = (mean_ == torch.max(mean_)).nonzero()
    #print(inx)
    mi = std_[inx[:,0],inx[:,1]]==torch.min(std_[inx[:,0],inx[:,1]])
    #print(mi,inx[mi])
    #print(mean_[inx[mi,0],inx[mi,1]],std_[inx[mi,0],inx[mi,1]])
    if(inx.size()[0]>1):
        print('max more than 1')
    max_x = inx[mi][0,1]
    max_y = inx[mi][0,0]
    if obj_ == None:
        return None,max_x,max_y,None,None
    i = 0
    pre_obj = []
    for x1, y1, x2, y2, conf, cls_conf, cls_pred in obj_: 
        x1_ = torch.ceil(x1).long()
        x2_ = torch.floor(x2).long()
        y1_ = torch.ceil(y1).long()
        y2_ = torch.floor(y2).long()
        if(max_x>x1_ and max_x<x2_ and max_y>y1_ and max_y<y2_):
            pre_obj.append(obj_[i])
            #break
        i += 1
    if(len(pre_obj) == 0):
        pre_obj = None
        return pre_obj,max_x,max_y,None,None
    else:
        pre_obj,area_max_mean,area_min_std = find_max_sum(mean_,std_,pre_obj)

    return pre_obj,max_x,max_y,area_max_mean,area_min_std

def sum_area(x1,x2,y1,y2,mean_,std_):
    if(x2 > mean_.size()[-1]):
        x2 = torch.tensor(mean_.size()[-1],dtype=torch.float32)
    if(x1 < 0):
        x1 = torch.tensor(0,dtype=torch.float32)
    if(y2 > mean_.size()[-2]):
        y2 = torch.tensor(mean_.size()[-2],dtype=torch.float32)
    if(y1 < 0):
        y1 = torch.tensor(0,dtype=torch.float32)
    #print(x1,x2,y1,y2)
    x = torch.arange(start=torch.ceil(x1),end=torch.floor(x2), step=1, out=None, dtype=torch.long)
    y = torch.arange(start=torch.ceil(y1),end=torch.floor(y2), step=1, out=None, dtype=torch.long)
    grid_y, grid_x = torch.meshgrid(y,x)
    area = (x2-x1)*(y2-y1)
    #print(grid_x,grid_y)
    sum_mean = torch.sum(mean_[grid_y,grid_x])/area
    sum_std  = torch.sum(std_[grid_y,grid_x])/area
    #print(sum_mean,sum_std)
    return sum_mean,sum_std

def find_max_sum(mean_,std_,obj_):
    #box_mid_x = torch.round((obj_[:,0]+obj_[:,2])/2).long()
    #box_mid_y = torch.round((obj_[:,1]+obj_[:,3])/2).long()
    all_max_mean = None
    all_min_std  = None
    i = 0
    obj_index = 0
    for x1, y1, x2, y2, conf, cls_conf, cls_pred in obj_: 
        sum_mean , sum_std = sum_area(x1,x2,y1,y2,mean_,std_)
        if(all_max_mean is None):
            all_max_mean = sum_mean
            all_min_std  = sum_std
            obj_index = 0
        else:
            if(all_max_mean<sum_mean):
                all_max_mean = sum_mean
                all_min_std  = sum_std
                obj_index = i
            elif(all_max_mean==sum_mean):
                if(all_min_std > sum_std):
                    all_max_mean = sum_mean
                    all_min_std  = sum_std
                    obj_index = i
        i += 1
        #inx = (std_[grid_y,grid_x]==min_std)
        # grid_y,grid_x = grid_y[inx],grid_x[inx]
        # print(min_std)
        # print(std_[grid_y,grid_x],grid_y,grid_x)
        #print(obj_index)
    return obj_[obj_index],all_max_mean,all_min_std

def draw_obj_box(obj_,ax,colors,classes,pre_obj=None,area_mean=False,mean_=None,std_=None):
    unique_labels = obj_[:, -1].unique()
    n_cls_preds = len(unique_labels)
    bbox_colors = colors
    if(pre_obj is None):
        pre_obj = obj_
    else:
        pre_obj = [pre_obj]
    i = 0
    for x1, y1, x2, y2, conf, cls_conf, cls_pred in pre_obj:
        #print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
        box_w = x2 - x1
        box_h = y2 - y1
        #color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
        color = bbox_colors[int(cls_pred)]
        # Create a Rectangle patch
        bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=1, edgecolor=color, facecolor="none")
        # Add the bbox to the plot
        ax.add_patch(bbox)
        # Add label
        if area_mean == True:
            sum_mean,sum_std = sum_area(x1,x2,y1,y2,mean_,std_)
            str_ = f'{classes[int(cls_pred)]}_C:{conf.item()*cls_conf.item():.2f}_S:{sum_mean:.3f}'
        else:
            str_ = f'{classes[int(cls_pred)]}_C:{conf.item()*cls_conf.item():.2f}'
        ax.text(
            x1,
            y1,
            s=str_,
            color="black",
            verticalalignment="top",
            bbox={"color": color, "pad": 0},
            fontsize=2
        )
        i += 1
    print('box :',i)

def save_img(save_path,img):
    img = np.round(img*255)
    cv2.imwrite(f'{save_path}.png',img)

def set_plt_img(fig,save_path,img_size_r=480,img_size_c=640):
    fig.set_size_inches(img_size_c/400.,img_size_r/400.)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
    plt.margins(0,0)
    fig.savefig(f"{save_path}.png", bbox_inches="tight", pad_inches=0.0,dpi=400)
    plt.close()

if __name__ == "__main__":
    torch.backends.cudnn.deterministic =True
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../example_img", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="./yoloatt_v3_split_rect.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="../weights/yoloatt_v3_split_w.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="./coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    #parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--out_path", type=str, default='./outputs',help="out_path")
    opt = parser.parse_args()
    print(opt)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(opt.out_path, exist_ok=True)

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index
    map_p_list = []
    std_p_list = []
    pre_obj = None

    in_shape_r,in_shape_c = 224,320
    #tar_shape_r,tar_shape_c = 480,640
    dataset = generator_SAL_test(opt.data_path,in_shape_r,in_shape_c)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=False, num_workers=8)
    print("\nPerforming object detection:")
    prev_time = time.time()
    init = 0
    print('load '+opt.weights_path)
    model = Darknet(opt.model_def).to(device)
    model.eval()
    model.load_state_dict(torch.load(opt.weights_path))
    for batch_i, (input_imgs,img_paths) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
        if(batch_i < init):
            continue
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            _,detections,outputs = model(input_imgs.to(device))
            #dect
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)
            #att
            out = outputs[-1][:,0,:,:]
            out = out.reshape(out.size()[0],1,out.size()[-2],out.size()[-1])
            std = outputs[-1][:,1,:,:]
            std = std.reshape(std.size()[0],1,std.size()[-2],std.size()[-1])
        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        #print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))
        imgs.extend(img_paths)
        img_detections.extend(detections)
        map_p_list.extend(list(out[:,...]))
        std_p_list.extend(list(std[:,...]))
        # Save image and detections
        #if(batch_i == 10+init):
        #    break
    # try:
    #     os._exit(0)
    # except:
    #     print('Program is dead.')
    # finally:
    #     print('clean-up')
    # Bounding-box colors

    color_dic = {'tab20c':20,'tab20b':20,'tab20':20,'Set2':8,'Set3':12}
    colors = []
    for k,v in color_dic.items():
        cmap = plt.get_cmap(k)
        colors_ = [cmap(i) for i in np.linspace(0, 1,v)]
        colors.extend(colors_)

    print("\nSaving images:")
    # Iterate through images and save plot of detections
    for img_i, (path,detections,map_p,std_p) in enumerate(zip(imgs,img_detections,map_p_list,std_p_list)):

        print("(%d) Image: '%s'" % (img_i, path))
        save_p = f"{opt.out_path}/ident/{path.split('.')[0]}"
        os.makedirs(save_p,exist_ok=True)
        path_ = f'{opt.data_path}/{path}'
        # Create plot
        
        img = np.array(Image.open(path_))
        tar_shape_r = img.shape[0]
        tar_shape_c = img.shape[1]
        map_p = nn.functional.interpolate(map_p.unsqueeze(0),size=[tar_shape_r,tar_shape_c],mode='bilinear').reshape(tar_shape_r,tar_shape_c)
        std_p = nn.functional.interpolate(std_p.unsqueeze(0),size=[tar_shape_r,tar_shape_c],mode='bilinear').reshape(tar_shape_r,tar_shape_c)

        map_ps = map_p.cpu().numpy().reshape(tar_shape_r,tar_shape_c)
        save_img(f'{save_p}/map_Pred',map_ps)
        cv2.imwrite(f'{save_p}/img.png',cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.axis('off')
        if detections is not None:
            # Rescale boxes to original image
            #detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
            w_factor = img.shape[1]/in_shape_c
            h_factor = img.shape[0]/in_shape_r
            detections[:,0] *= w_factor
            detections[:,2] *= w_factor
            detections[:,1] *= h_factor
            detections[:,3] *= h_factor

            pre_obj,max_x,max_y,area_max_mean,area_min_std = find_max(map_p,std_p,detections)
            draw_obj_box(detections,ax,colors,classes,area_mean=True,mean_=map_p,std_=std_p)
        else:
            _,max_x,max_y,_,_ = find_max(map_p,std_p,detections)
        set_plt_img(fig,f'{save_p}/yoloatt_obj_pred',img_size_r=img.shape[0],img_size_c=img.shape[1])
        
        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.axis('off')
        if(pre_obj is not None and detections is not None):
            draw_obj_box(detections,ax,colors,classes,pre_obj=pre_obj,area_mean=True,mean_=map_p,std_=std_p)
        set_plt_img(fig,f'{save_p}/yoloatt_salobj_pred',img_size_r=img.shape[0],img_size_c=img.shape[1])

        img_obj = np.array(Image.open(f'{save_p}/yoloatt_obj_pred.png'))
        img_salobj = np.array(Image.open(f'{save_p}/yoloatt_salobj_pred.png'))
        fig, ax = plt.subplots(ncols=4,num='Total')
        img_dic = {'img':img,'obj_pred':img_obj,'salobj_pred':img_salobj,'map_pred':map_p.cpu()}
        idx = 0
        for k,v in img_dic.items():
            ax[idx].axis('off')
            if('map' in k):
                ax[idx].imshow(v,cmap=plt.cm.gray)
                #ax[idx].plot(max_x.cpu(),max_y.cpu(),'ro',markersize=1)
            else:
                ax[idx].imshow(v)
            ax[idx].set_title(k,fontdict={'fontsize':4})
            idx += 1
        set_plt_img(fig,f"{opt.out_path}/{path.split('.')[0]}",img_size_r=img.shape[0]*idx,img_size_c=img.shape[1]*idx)
