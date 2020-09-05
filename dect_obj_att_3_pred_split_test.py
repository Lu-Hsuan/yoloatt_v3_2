from __future__ import division

from models_rect import *
from utils.utils import *
from utils.datasets import *
from att_dataset import generator_SAL_metric

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

def save_img(save_path,img):
    img = np.round(img*255)
    cv2.imwrite(f'{save_path}.png',img)

def set_plt_img(fig,save_path):
    fig.set_size_inches(640/400.,480/400.)
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

    in_shape_r,in_shape_c = 224,320
    tar_shape_r,tar_shape_c = 480,640
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
    dataset = generator_SAL_metric(opt.data_path,"val",in_shape_r,in_shape_c,file_list='../data/common_sal.txt')
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=False, num_workers=8)
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
            logits = nn.functional.interpolate(out,size=[tar_shape_r,tar_shape_c],mode='bilinear')
            map_p = logits.cpu().numpy().reshape(opt.batch_size,tar_shape_r,tar_shape_c)
        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        #print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))
        imgs.extend(img_paths)
        img_detections.extend(detections)
        map_p_list.extend(list(map_p[:,...]))
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
    for img_i, (path,detections,map_p) in enumerate(zip(imgs,img_detections,map_p_list)):

        print("(%d) Image: '%s'" % (img_i, path))
        save_p = f"{opt.out_path}/ident/{path.split('.')[0]}"
        os.makedirs(save_p,exist_ok=True)
        path = f'{data_path}/{path}'
        # Create plot

        img = np.array(Image.open(path))
        save_img(f'{save_p}/map_Pred',map_p)
        cv2.imwrite(f'{save_p}/img.png',cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        #fig, ax = plt.subplots(ncols=5,num='Total')
        # Draw bounding boxes and labels of detections
        #"""
        
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
            print(detections)
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = colors
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

                box_w = x2 - x1
                box_h = y2 - y1
                #color = bbox_colors[int(cls_pred)]
                color = bbox_colors[int(cls_pred)]
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=1, edgecolor=color, facecolor="none")
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                ax.text(
                    x1,
                    y1,
                    s=classes[int(cls_pred)],
                    color="black",
                    verticalalignment="top",
                    bbox={"color": color, "pad": 0},
                    fontsize=2
                )
        set_plt_img(fig,f'{save_p}/yoloatt_obj_pred')
        