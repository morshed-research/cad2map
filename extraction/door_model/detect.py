import sys
sys.path.append('')

import numpy as np
import cv2
import os
from datetime import datetime

from PIL import Image

import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import pandas as pd

root = os.path.dirname(__file__) + "/../"
device = torch.device("cuda" 
    if torch.cuda.is_available() else "mps" 
    if torch.backends.mps.is_available() and torch.backends.mps.is_built()
    else "cpu"
)

"""
applies transformations for image to be passed into detection model

parameters
- img: image to pass to model, cv2 image
    (required)

returns 
    transformed image, numpy array
"""
def img_transform(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img /= 255.0
    img = torch.from_numpy(img).permute(2,0,1)
    return img

"""
Infernece of a single input image

parameters
- img: input-image as torch.tensor (shape: [C, H, W])
    (required)
- model: model for infernce (torch.nn.Module)
    (required)
- detection_threshold: Confidence-threshold for NMS 
    (default 0.7)

returns
    boxes: bounding boxes (Format [N, 4] => N times [xmin, ymin, xmax, ymax])
    labels: class-prediction (Format [N] => N times an number between 0 and _num_classes-1)
    scores: confidence-score (Format [N] => N times confidence-score between 0 and 1)
"""
def inference(img, model, detection_threshold=0.00):
    model.eval()

    img = img.to(device)
    outputs = model([img])
    
    boxes = outputs[0]['boxes'].data.cpu().numpy()
    scores = outputs[0]['scores'].data.cpu().numpy()
    labels = outputs[0]['labels'].data.cpu().numpy()

    boxes = boxes[scores >= detection_threshold].astype(np.int32)
    labels = labels[scores >= detection_threshold]
    scores = scores[scores >= detection_threshold]

    return boxes, scores, labels

"""
Function that draws the BBoxes, scores, and labels on the image.

inputs:
  img: input-image as numpy.array (shape: [H, W, C])
  boxes: list of bounding boxes (Format [N, 4] => N times [xmin, ymin, xmax, ymax])
  scores: list of conf-scores (Format [N] => N times confidence-score between 0 and 1)
  labels: list of class-prediction (Format [N] => N times an number between 0 and _num_classes-1)
  dataset: list of all classes e.g. ["background", "class1", "class2", ..., "classN"] => Format [N_classes]
"""
def plot_image(img, boxes, scores, labels, dataset, save_path=None):
    cmap = plt.get_cmap("tab20b")
    class_labels = np.array(dataset)

    colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
    height, width, _ = img.shape
    
    # Create figure and axes
    fig, ax = plt.subplots(1, figsize=(16, 8))
    
    # Display the image
    ax.imshow(img)
    
    for i, box in enumerate(boxes):
      class_pred = labels[i]
      conf = scores[i]
      
      width = box[2] - box[0]
      height = box[3] - box[1]
      
      rect = patches.Rectangle(
          (box[0], box[1]),
          width,
          height,
          linewidth=2,
          edgecolor=colors[int(class_pred)],
          facecolor="none",
      )
      
      # Add the patch to the Axes
      ax.add_patch(rect)
      plt.text(
          box[0], box[1],
          s=class_labels[int(class_pred)] + " " + str(int(100*conf)) + "%",
          color="white",
          verticalalignment="top",
          bbox={"color": colors[int(class_pred)], "pad": 0},
      )

    # Used to save inference phase results
    if save_path is not None:
      plt.savefig(save_path)

    plt.show()

"""
sets up detection model to perform bounding box detection with

parameters
- weights_pth: location to pretrained model weights, str
    (required)

returns
    FasterRCNN model, from pytorch
"""
def detection_model(weights_pth):
    num_classes = 2  # 1 classes + background

    # set up FasterRCNN model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(min_size=300,
        max_size=480, weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)

    # load pretrained weights
    checkpoint = torch.load(weights_pth, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model