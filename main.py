from sys import argv
from time import time
import os

t = time()
import pandas as pd

from utils.graph import graph

from slicing.image_slice import slice_images

from extraction.area import get_rooms, make_nodes
from extraction.door import door_boxes, make_doors
from extraction.merge import scale_nodes, merge_labels, merge_dist

from linking.edges import door_edges

from completion.edges import door_to_connective, radial_edges

from eval.testbed import test_doors

print(f"imports: {time() - t}")
argc = len(argv)

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
    # generate image slice panels
    if slice_csv != "":
        print(argv[2])
        panels_df = pd.read_csv(argv[2])
    else:
        if not os.path.exists("data/test-panels/"):
            os.makedirs("data/test-panels/")
        panels_df = slice_images(image_name, "data/",
                          0.5, 1000,"data/test-panels/")
        
    G = graph()

    # get all area and door nodes for each slice
    for i in range(len(panels_df)):
        row = panels_df.iloc[i]
        image_path = row["name"]
        print(image_path)

        df = get_rooms(image_path)
        if df.empty:
            continue
        
        local = make_nodes(df)
        local = merge_labels(local) # label based merging

        # get doors
        door_df = door_boxes(image_path, thresh=0.5)
        local = make_doors(door_df, local)

        # add to global graph with corrected coordinate position
        G = scale_nodes(local, G, int(row.xmin), int(row.ymin))

    # group merge across all slices
    G = merge_dist(G)
    G.set_connective() # mark connective areas by label names

    # create edges through doors
    G = door_edges(G)
    G = door_to_connective(G)

    # connect between connective areas
    G = radial_edges(G, n=2)

    # save JSON and visualisation
    name = image_name.removesuffix('.jpeg').removesuffix(".png").removesuffix(".jpg").replace("/", "-")
    G.draw(f"data/{image_name}", 
           f"results/{name}-graph.png", label_it=False)
    
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