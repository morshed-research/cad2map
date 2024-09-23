import os
from sys import argv
import pandas as pd
from tqdm import tqdm
import cv2 

from time import time
t = time()

from utils.graph import graph

from slicing.image_slice import slice_images
from extraction.area import get_rooms, make_nodes
from extraction.door import door_boxes, make_doors
from extraction.door_model.detect import detection_model
from extraction.label_model.read import prep_read_labels
from extraction.label_model.bboxes import bbox_parser, set_model
from extraction.segment import segment_areas_fill, segment_hallways_fill, add_hallways_graph
from extraction.merge import scale_nodes, merge_labels, merge_by_dist
from linking.edges import door_edges
from completion.edges import door_to_connective
from eval.testbed import test_doors

print(f"imports: {time() - t}")
argc = len(argv)
root = os.path.dirname(__file__)

"""
applies full mapmaking pipeline on given image

parameters
- image_name: path of image under data folder, including filename, str
    (required)
- slice_csv: csv containing info on all image slices precomputed and saved, str
    (default is slices generate from scratch)

returns
    path to graph's JSON file, str
"""
def pipeline(image_name, slice_csv=""):
    org_image_path = f'data/{image_name}'

    # generate image slice panels
    print("Slicing Input Image...")
    if slice_csv != "":
        print(argv[2])
        panels_df = pd.read_csv(argv[2])
    else:
        if not os.path.exists("data/test-panels/"):
            os.makedirs("data/test-panels/")
        panels_df = slice_images(image_name, "data/",
                          0.5, 1000,"data/test-panels/")
    
    G = graph()

    # Setup text detection and interpreting models
    detect_model_args = bbox_parser(root+"/model_weights/craft_mlt_25k.pth")
    detect_model, _ = set_model(detect_model_args)
    interp_model = prep_read_labels("None","VGG","BiLSTM","CTC", root+"/model_weights/None-VGG-BiLSTM-CTC.pth")
    door_model = detection_model(root+'/model_weights/door_mdl_32.pth')

    # get all area and door nodes for each slice
    for i in tqdm(range(len(panels_df)), 'Getting Area Nodes and Door Node...'):
        row = panels_df.iloc[i]
        image_path = row["name"]

        df = get_rooms(image_path, detect_model_args, detect_model, interp_model)
        # if not df.empty:
        #     continue
        
        local_G = make_nodes(df)
        local_G = merge_labels(local_G) # label based merging

        # get doors
        door_df = door_boxes(image_path, door_model, thresh=0.5)
        local_G = make_doors(door_df, local_G)

        # add to global graph with corrected coordinate position
        G = scale_nodes(local_G, G, int(row.xmin), int(row.ymin))

    # group merge doors across all slices
    G = merge_by_dist(G)
    
    # Create connective areas graph
    area_segments = segment_areas_fill(G, org_image_path)
    G, hallway_segments = segment_hallways_fill(G, org_image_path, area_segments)
    G, hallways_node2pixel = add_hallways_graph(G, org_image_path, hallway_segments)

    # Connect doors to connective areas
    G = door_to_connective(G)

    # Connect doors to their rooms
    # G = door_edges(G)

    # save JSON and visualisation
    name = (
        image_name
            .removesuffix('.jpeg')
            .removesuffix(".png")
            .removesuffix(".jpg")
            .replace("/", "-")
        )
    G.draw(org_image_path, area_segments, hallway_segments, True,  
           f"results/{name}-graph.png")
    
    G.to_json(f"results/json/{name}-graph.json")

    print(f"total run time:{time() - t}secs")
    return f"results/json/{name}-graph.json"


if __name__ == '__main__':
    if argc != 2 and argc != 3:
        print("usage: python main.py 'image_name'")
        print("alternate usage: python main.py 'image_name' 'slice csv'")
        exit(1)

    image_name = argv[1]

    if argc == 3:
        print(argv[2])
        pipeline(image_name, argv[2])
    else:
        pipeline(image_name)