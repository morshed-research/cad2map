from utils.graph import *
from .door_model.detect import detection_model, img_transform, inference

import pandas as pd
import os
root = os.path.dirname(__file__) + "/../"

"""
creates door nodes based on given door bounding boxes and adds them to 
the given graph

parameters
- door_df: dataframe with door bounding boxes with 
           columns 'xmin' 'ymin' 'xmax' 'ymax', pd.DataFrame
    (required)
- G: graph to add nodes to, graph
    (required)

returns 
    graph with door nodes added, graph
"""
def make_doors(door_df, G):
    for i, row in door_df.iterrows():
        x1, y1, x2, y2 = row["xmin"], row["ymin"], row["xmax"], row["ymax"]

        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        G.add_node(node(id=G.next_id, area="Door", coord=center, type="door"))

    return G

"""
performs inference on image at given location and returns a dataframe of all found doors

parameters
- image_path: location of image to infer on, str
    (required)
- weights_pth: location to pretrained model weights, str
    (default '../model_weights/door_mdl_32.pth')
- thresh: threshold of confidence to consider inference valid, float
    (default 0.0)

returns 
    dataframe of door bounding boxes with columns 
    'xmin' 'ymin' 'xmax' 'ymax', pd.DataFrame
"""
def door_boxes(image_path, weights_pth=root + "model_weights/door_mdl_32.pth", 
               thresh=0):
    model = detection_model(weights_pth)

    # transform and infer
    img = img_transform(cv2.imread(image_path))
    boxes, scores, labels = inference(img, model, detection_threshold=thresh)

    # make dataframe from found boxes
    return pd.DataFrame(boxes, columns=["xmin","ymin","xmax","ymax"])