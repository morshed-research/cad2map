import sys
import os
sys.path.append("../")

from .plan import Room, Door, Plan, mean 
from .metrics import Path_Similarity
from .testbed import to_graph, to_IL_graph
from utils.graph import node, graph

from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFont
import gmatch4py as gm 

from main import pipeline

connective = ["LivingRoom", "Entry"] # add labels to consider connective here

"""
given a path to a folder containing a floorplan image and equivalent SVG,
generates an annotated image to pass to the mapmaking pipeline

parameters
- img_path: folder that contains model.svg and F1_scaled.png, not ending in '/', str
    (required)

returns
    None, image saved under same folder as 'F1_scaled_labelled.png'

"""
def input_image(img_path):
    # read SVG floorplan
    with open(f"{img_path}/model.svg") as f:
        content = f.read()
    soup = BeautifulSoup(content, 'lxml')

    # Create a Plan instance using SVG
    plan = Plan(soup)

    # get base image
    img = Image.open(f"{img_path}/F1_scaled.png")
    font = ImageFont.truetype("/home/nsl/Documents/cad2map/eval/Arial.ttf", 50)
    plot = ImageDraw.Draw(img)

    # add labels for each room at its centre
    for room in plan.rooms:
        position = room.center_point
        bbox = plot.textbbox(position, room.name, font=font)

        plot.rectangle(bbox, fill="white") # fill with white background box
        plot.text(position, room.name, font=font, fill="black")

    img.save(f"{img_path}/F1_scaled_labelled.png")

"""
given a path to a folder containing a floorplan image and equivalent SVG,
loads the CubiGraph5k ground truth as a graph object

parameters
- img_path: folder that contains model.svg and F1_scaled.png, not ending in '/', str
    (required)

returns
    ground truth graph object, graph
"""
def load_gt_graph(img_path):
    G = graph()

    # read SVG floorplan
    with open(f"{img_path}/model.svg") as f:
        content = f.read()
    soup = BeautifulSoup(content, 'lxml')

    # Create a Plan instance using SVG
    plan = Plan(soup)
    name2node = dict()

    # add room nodes
    for room in plan.rooms:
        position = room.center_point # associate with centre of room
        node_type = "connective" if room.svg_class in connective else "area" # assign type
        
        # add node to graph
        n = node(G.next_id, room.name, position, node_type)
        G.add_node(n)

        # maintain mapping by unique name identifier
        name2node[room.name] = n

    # add door nodes
    for door in plan.doors:
        position = door.center_point # associate with center of door

        # add door node
        d = node(id=G.next_id, area="Door", coord=position, type="door")
        G.add_node(d)

        # maintain mapping by unique door identifier
        name2node[door.name] = d 

    # compute all overlapping doors
    for room in plan.rooms:
            room.get_adjacent_doors(plan.doors)
    
    # extract all edges
    for i in range(len(plan.rooms)):
        for j in range(i+1, len(plan.rooms)):
            room1 = plan.rooms[i]
            room2 = plan.rooms[j]

            # get commonly intersecting doors
            door_intersect = room1.adjacent_doors.intersection(room2.adjacent_doors)

            if room1.to_shapely_polygon().buffer(1.0).intersection(room2.to_shapely_polygon().buffer(1.0)).area > 5.0:
                # room to room relation
                G.nx_graph.add_edge(name2node[room1.name], name2node[room2.name])
            elif door_intersect: # connection through doors
                for door_name in door_intersect:
                    # add edge to door node from both sides (i.e. both rooms)
                    G.nx_graph.add_edge(name2node[door_name], name2node[room1.name])
                    G.nx_graph.add_edge(name2node[door_name], name2node[room2.name])

    return G

# sample engine
if __name__ == "main":
    root = os.path.dirname(__file__) + "/../"
    file = open(root + "data/cubicasa5k/all.txt") # read all floorplans
    results = dict()

    for line in file:
        folder = line.strip()

        input_image(folder) # generate annotations
        name = pipeline(f"{folder}/F1_scaled_labelled.png") # pass annotations

        # start evaluation
        target = load_gt_graph(folder)

        file = open(name, "r")
        source = to_graph(file)
        file.close()

        # get both metric results
        metric = Path_Similarity(source, target)
        nx1, nx2 = to_IL_graph(target, source)
        ged = gm.GraphEditDistance(1, 1, 1, 1) # all edit costs are equal to 1
        result = ged.compare([nx1, nx2], None) 

        # save results as sub dictionary
        results[folder] = {"path": metric.evaluate(), "GED": result}

    # you could now load results into a JSON below this line...