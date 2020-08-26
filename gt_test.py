from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="../data/coco/5k.txt", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="./yoloatt_v3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="../weights/yoloatt_v3_2_w.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="./coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--out_path", type=str, default='./outputs',help="out_path")
    opt = parser.parse_args()
    print(opt)

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index
    target_list = []
    dataset = ListDataset(opt.image_folder, img_size=opt.img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=False, num_workers=2, collate_fn=dataset.collate_fn
    )
    os.makedirs(opt.out_path, exist_ok=True)
    print("\nPerforming object detection:")
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
        anno = []
        for sample_i in range(opt.batch_size):
            annotations = targets[targets[:, 0] == sample_i][:, 1:]
            #target_labels = annotations[:, 0] if len(annotations) else []
            anno.append(annotations)
        imgs.extend(img_paths)
        target_list.extend(anno)
        if(batch_i == 1):
            break

    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    print("\nSaving images:")
    # Iterate through images and save plot of detections
    for img_i, (path,target_gt) in enumerate(zip(imgs,target_list)):
        print("(%d) Image: '%s'" % (img_i, path))
        # Create plot
        img = np.array(Image.open(path))
        plt.figure()
        img_ = transforms.ToTensor()(img)
        img_, pad = pad_to_square(img_, 0)
        #if Padding
        #img_ = resize(img_,416)
        #print(img_.size())
        #img = np.array(transforms.ToPILImage()(img_))
        fig, ax = plt.subplots(ncols=2)
        ax[0].imshow(img)
        ax[1].imshow(img)
        ax[0].axis("off")
        ax[1].axis("off")
        # Draw bounding boxes and labels of detections
        if target_gt is not None:
            # Rescale boxes to original image
            #print(len(target_gt))
            target_gt[:,1:] = xywh2xyxy(target_gt[:,1:])
            #'''
            ''' if Padding
            target_gt[:,1] *= img.shape[1]
            target_gt[:,3] *= img.shape[1]
            target_gt[:,2] *= img.shape[0]
            target_gt[:,4] *= img.shape[0]
            '''
            #'''NOT Padding
            target_gt[:,1] *= img_.shape[2]
            target_gt[:,3] *= img_.shape[2]
            target_gt[:,2] *= img_.shape[1]
            target_gt[:,4] *= img_.shape[1]

            target_gt[:,1] = (target_gt[:,1] - pad[0])
            target_gt[:,3] = (target_gt[:,3] - pad[0])
            target_gt[:,2] = (target_gt[:,2] - pad[2])
            target_gt[:,4] = (target_gt[:,4] - pad[2])
            #'''
            #target_gt[:,1:] = rescale_boxes(target_gt[:,1:], opt.img_size, img.shape[:2])
            unique_labels = target_gt[:,0].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            print(target_gt)
            for cls_pred,x1, y1, x2, y2 in target_gt:

                #print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

                box_w = x2 - x1
                box_h = y2 - y1

                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=1, edgecolor=color, facecolor="none")
                # Add the bbox to the plot
                ax[0].add_patch(bbox)
                # Add label
                ax[0].text(
                    x1,
                    y1,
                    s=classes[int(cls_pred)],
                    color="white",
                    verticalalignment="top",
                    bbox={"color": color, "pad": 0},
                    fontsize=2
                )
        
        # Save generated image with detections
        
        #plt.gca().xaxis.set_major_locator(NullLocator())
        #plt.gca().yaxis.set_major_locator(NullLocator())
        filename = path.split("/")[-1].split(".")[0]
        fig.set_size_inches(1200/100.0/4.0, 800/100.0/4.0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
        plt.margins(0,0)
        fig.savefig(f"{opt.out_path}/{filename}.png", bbox_inches="tight", pad_inches=0.0,dpi=400)
        plt.close()