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
    parser.add_argument("--model_def", type=str, default="./yoloatt_v3_split.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="../weights/yoloatt_v3_split_w.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="./coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--out_path", type=str, default='./outputs',help="out_path")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_s_path = f'{opt.out_path}/img'
    tar_s_path = f'{opt.out_path}/tar'
    det_s_path = f'{opt.out_path}/det'
    os.makedirs(img_s_path, exist_ok=True)
    os.makedirs(tar_s_path, exist_ok=True)
    os.makedirs(det_s_path, exist_ok=True)

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  # Set in evaluation mode

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index
    target_list = []
    dataset = ListDataset(opt.image_folder, img_size=opt.img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=False, num_workers=8, collate_fn=dataset.collate_fn
    )

    print("\nPerforming object detection:")
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
    #for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            _,detections,_ = model(input_imgs.to(device),targets.to(device))
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)
            print(f'loss:{_}')
        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        #print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))
        imgs.extend(img_paths)
        img_detections.extend(detections)
        # Save image and detections
        if(batch_i == 5):
            break

    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    print("\nSaving images:")
    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        print("(%d) Image: '%s'" % (img_i, path))

        # Create plot
        img = np.array(Image.open(path))
        label_path = path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
        #print(label_path)
        target_gt = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
        #print(target_gt)
        plt.figure()
        fig, ax = plt.subplots(ncols=2)
        ax[0].imshow(img)
        ax[1].imshow(img)
        ax[0].axis("off")
        ax[1].axis("off")
        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
            print(detections)
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

                box_w = x2 - x1
                box_h = y2 - y1

                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=1, edgecolor=color, facecolor="none")
                # Add the bbox to the plot
                ax[1].add_patch(bbox)
                # Add label
                ax[1].text(
                    x1,
                    y1,
                    s=classes[int(cls_pred)],
                    color="white",
                    verticalalignment="top",
                    bbox={"color": color, "pad": 0},
                    fontsize=2
                )
        if target_gt is not None:
            # Rescale boxes to original image
            #print(len(target_gt))
            #target_gt[:,1:] = xywh2xyxy(target_gt[:,1:])
            #'''
            w_factor = img.shape[1]
            h_factor = img.shape[0]
            x1 = w_factor * (target_gt[:, 1] - target_gt[:, 3] / 2)
            y1 = h_factor * (target_gt[:, 2] - target_gt[:, 4] / 2)
            x2 = w_factor * (target_gt[:, 1] + target_gt[:, 3] / 2)
            y2 = h_factor * (target_gt[:, 2] + target_gt[:, 4] / 2)
            target_gt[:,1] = x1
            target_gt[:,3] = x2
            target_gt[:,2] = y1
            target_gt[:,4] = y2
            #'''
            #target_gt[:,1:] = rescale_boxes(target_gt[:,1:], opt.img_size, img.shape[:2])
            unique_labels = target_gt[:,0].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            #print(target_gt)
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
                #'''
                ax[0].text(
                    x1,
                    y1,
                    s=classes[int(cls_pred)],
                    color="white",
                    verticalalignment="top",
                    bbox={"color": color, "pad": 0},
                    fontsize=2
                )
                #'''
        
        # Save generated image with detections
        
        #plt.gca().xaxis.set_major_locator(NullLocator())
        #plt.gca().yaxis.set_major_locator(NullLocator())
        filename = path.split("/")[-1].split(".")[0]
        if(target_gt is not None):
            target_gt = target_gt.cpu().numpy()
            np.save(f'{tar_s_path}/{filename}.npy',target_gt)
        if(detections is not None):
            detections = detections.cpu().numpy()
            np.save(f'{det_s_path}/{filename}.npy',detections)
        fig.set_size_inches(1200/100.0/4.0, 800/100.0/4.0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
        plt.margins(0,0)
        fig.savefig(f"{img_s_path}/{filename}.png", bbox_inches="tight", pad_inches=0.0,dpi=400)
        plt.close()