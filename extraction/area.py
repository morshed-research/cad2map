import sys
sys.path.append('../')

import pandas as pd
import os
import cv2

from .label_model.bboxes import get_bboxes
from .label_model.crop_image import generate_image
from .label_model.read import prep_read_labels, interpret_labels

from utils.graph import *

root = os.path.dirname(__file__) + "/../"

"""
performs bounding box detection and text inference on given image and
returns results

parameters
- image_path: location of image to detect on, str
    (required)
- result_folder: folder to save any intermediate plots, str
    (default '../results/')
- weights: CRAFT pretrained weights location, str
    (default '../model_weights/craft_mlt_25k.pth')
- vgg: VGG BiLSTM weights location, str
    (default '../model_weights/None-VGG-BiLSTM-CTC.pth')

returns
    dataframe with columnx 'bbox' 'pred_words' 'confidence_score', pd.DataFrame
"""
def get_rooms(image_path: str, 
              result_folder: str=root + "results/",
              weights=root + "model_weights/craft_mlt_25k.pth",
              vgg=root + "model_weights/None-VGG-BiLSTM-CTC.pth"):
    # make result folder if doesn't exist
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)
    
    # get all text bounding boxes
    bboxes = get_bboxes(image_path, weights, result_folder)
    
    # open image as cv2 image
    image = cv2.imread(image_path)

    # set up arguments for model
    opt = prep_read_labels("None","VGG","BiLSTM","CTC", vgg)
    
    df=pd.DataFrame(columns=['bbox', 'pred_words', 'confidence_score'])

    for i in range(len(bboxes)):
        # crop to label only
        img = generate_image(bboxes[i], image)

        if img == None: # failed to crop, e.g. empty box
            print(f"failed on box {i}/{len(bboxes)}\n")
        else:
            pred, score = interpret_labels(opt, img) # get label inference
            df.loc[i] = [bboxes[i], pred, score]

    return df.reset_index(drop=True)

"""
creates nodes based on all bounding boxes in the input dataframe

parameters
- df: dataframe of bounding boxes and their labels, pd.DataFrame
      with columns 'bbox' 'pred_words' 'confidence_score'
    (required)

returns 
    graph with each box as a node, graph
"""
def make_nodes(df):
    G = graph() # create empty graph

    for i in range(len(df)):
        coords = df['bbox'][i] # get each bounding box

        # min and max coordinates
        (x1, y1) = coords[0]
        (x2, y2) = coords[2]

        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        label = df['pred_words'][i]

        # correct mistaken inference
        if label.lower() in "iill":
            label = "hallway"

        # add as area node
        G.add_node(node(id=G.next_id, area=label, coord=center))

    return G