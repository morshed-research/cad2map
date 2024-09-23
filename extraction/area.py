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
def get_rooms(image_path: str, detect_model_args, detect_model, interp_model, 
              result_folder: str=root + "results/",):
    
    # make result folder if doesn't exist
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)
    
    # open image as cv2 image and create DataFame
    image = cv2.imread(image_path)
    df=pd.DataFrame(columns=['bbox', 'pred_words', 'confidence_score'])

    # get all text bounding boxes
    bboxes = get_bboxes(image_path, detect_model_args, detect_model, result_folder)

    # Detect text within each bounding box
    failed_labels = []
    for i in range(len(bboxes)):
        # crop to label only
        img = generate_image(bboxes[i], image)

        if img == None: # failed to crop, e.g. empty box
            failed_labels.append(bboxes[i])
            # print(f"failed on box {i}/{len(bboxes)}\n")
        else:
            pred, score = interpret_labels(interp_model, img) # get label inference
            df.loc[i] = [bboxes[i], pred, score]

    # if failed_labels:
        # print('Failed Labels:', failed_labels)
    
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
        (x1, y1) = coords[0] # bottom-left
        (x2, y2) = coords[2] # top-right

        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        label = df['pred_words'][i].lower()

        # add as area node
        node_i = node(id=G.next_id)
        node_i.set_coordinates(center[0], center[1])
        node_i.set_label(label)
        node_i.set_bbox(coords[0], coords[1])
        G.add_node(node_i)

    return G