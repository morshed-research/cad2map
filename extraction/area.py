import sys
sys.path.append('../')

import networkx as nx 
import pandas as pd
import os
import cv2
import math

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from .labels.bboxes import get_bboxes
from .labels.crop_image import generate_image
from .labels.read import prep_read_labels, interpret_labels
import extraction.labels.modules.file_utils as file_utils

from utils.graph import *
from utils.distance import *
from utils.connective import label_list

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

    return df

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

"""
returns the larger string out of the two given strings

parameters
- s1: first string to compare, str
    (required)
- s2: second string to compare, str
    (required)

returns
    string with longer length, str
"""
def select_label(s1, s2):
    if len(s1) > len(s2):
        return s1 
    else:
        return s2
    
"""
merges nodes which represent the same space/object in our graph's map

parameters
- G: graph to operate on, graph
- thresh: euclidean distance threshold to merge based on, int

returns
    updated graph, graph
"""
def merge_thresh_nodes(G, thresh):
    nodes = set(G.nx_graph.nodes) # set of all nodes
    new_G = graph() # new graph with merged nodes

    for n1 in G.nx_graph: # go over all nodes
        if n1 not in nodes: # if removed from set, already merged
            continue
        
        # add nodes that will be merged, you could only be 'merged' to yourself
        rm = set()
        rm.add(n1)
        
        # go over all nodes you could merge to
        for n2 in nodes:
            if n1 != n2 and n1.type == n2.type:
                dist, center = node_dist(n1, n2)

                # only merge nodes if euclidean distance under threshold
                if dist <= thresh:
                    n1.coordinates = center
                    n1.area_label = select_label(n1.area_label, n2.area_label)

                    rm.add(n2) 

        new_G.add_node(n1)

        #remove all merged nodes
        nodes = nodes.difference(rm)

    # update id accordingly, so new nodes added after even merged ones
    new_G.next_id = G.next_id
    return new_G

"""
merges all nodes that are within 2 metres in the given graph

parameters
- G: graph to merge nodes in, graph
    (required)

returns 
    graph with nodes merged, graph
"""
def merge_dist(G):
    two_metres = dist2pixel(2)
    return merge_thresh_nodes(G, two_metres)

"""
merges nodes if their labels have semantic meaning when combined

parameters
- G: graph to merge nodes in, graph
    (required)

returns 
    graph with nodes merged, graph
"""
def merge_labels(G):
    nodes = set(G.nx_graph.nodes) # set of all nodes
    new_G = graph() # new graph with merged nodes

    for n1 in G.nx_graph: # go over all nodes
        if n1 not in nodes: # if removed from set, already merged
            continue
        
        # add nodes that will be merged, you could only be 'merged' to yourself
        rm = set()
        rm.add(n1)
        
        # go over all nodes you could merge to
        for n2 in nodes:
            if n1 != n2 and n1.type == n2.type:
                dist, center = node_dist(n1, n2)

                # only merge nodes if combined label in connective list
                new_label = n1.area_label + n2.area_label
                if new_label in label_list().all:
                    n1.coordinates = center
                    n1.area_label = select_label(n1.area_label, n2.area_label)

                    rm.add(n2) 

        new_G.add_node(n1)

        #remove all merged nodes
        nodes = nodes.difference(rm)

    # update id accordingly, so new nodes added after even merged ones
    new_G.next_id = G.next_id
    return new_G