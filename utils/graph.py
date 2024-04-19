import sys
sys.path.append('../')

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import json

from .connective import label_list
from .distance import coord_match

node_area_types = ["area", "door", "connective"]
conn_labels = set(label_list().connective)

"""
Class for representing a node in our graph

properties
- node_id: id of node in the graph, int
- area_label: semantic name for node in graph's map, str
- coordinates: coordinate of node in graph's map, int * int
- type: type of node out of area, door and connective, str

create instance 
- node(id, area: default N/A, coordinate: default (-1,-1), type: default 'area')
"""
class node():
    """
    create node object with given parameters or default values

    parameters
    - id: id of node, int 
        (required)
    - area: semantic name, str 
        (default 'N/A')
    - coord: coordinate location of node, int * int 
        (default (-1, -1))
    - type: node type, str
        (default 'area')

    return
        node object with properties passed in
    """
    def __init__(self, id: int, area: str="N/A", coord=(-1,1), type: str="area"):
        self.node_id = id # id in graph
        self.area_label = area # label from annotation
        self.coordinates = coord # center coordinates (x, y)

        self.type = type # area or door node?

        self.test = False

    """
    update area_label property of node

    parameters
    - label: semantic name, str
        (required)

    returns
        None
    """
    def set_label(self, label: str):
        if not isinstance(label, str): # must be string
            raise TypeError
        else:
            self.area_label = label

    """
    update coordinates property of node

    parameters
    - x: x coordinate of node, int or numpy int types
    - y: y coordinate of node, int or numpy int types

    returns
        None
    """
    def set_coordinates(self, x: int, y: int):
        self.coordinates = (x, y)

    """
    update area_type property of node

    parameters
    - area_type: new type of node, str

    returns
        None
    """
    def set_type(self, area_type: str):
        if area_type not in node_area_types: # must be one of 3 valid types
            raise Exception("invalid area type for node\n")
        else:
            self.type = area_type

    """
    equivalence based on id upon creation

    invoked by ==
    """
    def __eq__(self, other):
        if isinstance(other, node) == False:
            return False
        
        if self.test:
            if self.type != other.type:
                return False
            # only area needs label match
            elif self.type == "door" or self.type == "connective":
                return coord_match(self, other)
            else:
                return self.area_label == other.area_label and coord_match(self, other)
        else:
            return self.node_id == other.node_id
    
    """
    hash based on unique id

    used in sets, dictionaries etc.
    """
    def __hash__(self):
        return hash(self.node_id)
    
    """
    print format: 'label: (x, y)'

    invoked by str(), print() etc.
    """
    def __repr__(self):
        return f"{self.area_label}: ({self.coordinates[0]},{self.coordinates[1]})"

"""
Class for representing our graph (for our map)

properties
- networkx graph: underlying graph structure to store nodes & edges
- next_id: global id count for adding nodes, int

create instance
- graph()
"""
class graph():
    """
    create node object with given parameters or default values

    parameters
        none

    returns 
        graph object instance with next_id = 0
    """
    def __init__(self):
        self.nx_graph = nx.Graph()
        self.next_id = 0

    """
    add a node to the graph

    parameters
    - n : node to add, node object
        (required)
    
    returns 
        None
    """
    def add_node(self, n):
        self.nx_graph.add_node(n)
        self.next_id += 1

    """
    draws the graph on top of the input image

    parameters
    - image_path: location of input image, str
        (required)
    - save_path: where to save resulting plot, str
        (required)
    - label_it: include node area_labels on plot, bool
        (default True)

    returns
        None
    """
    def draw(self, image_path, save_path, label_it=True):
        pos = {} # node coordinates
        colour = [] # node colours
        labeldict = {} # node labels

        for n in self.nx_graph: # get info for every node
            (x, y) = n.coordinates
            pos[n] = [x, y]
            labeldict[n] = n.area_label

            # differentiate door and other nodes
            if n.type == "door":
                colour.append("red")
            elif n.type == "connective":
                colour.append("orange")
            else:
                colour.append("blue")

        # create figure window
        plt.figure(figsize=(12.8, 9.6))
        img=mpimg.imread(image_path)
        plt.imshow(img) # apply base image

        options = {"node_size": 75, "node_color": colour}

        if label_it: # plot graph with labels
            nx.draw(self.nx_graph,pos, labels=labeldict, with_labels=True, **options)
        else: # plot graph without labels
            nx.draw(self.nx_graph,pos, with_labels=False, **options)
        plt.savefig(save_path) # save resulting plot

    """
    identifies all nodes that are connective areas by label and
    changes their type in the graph 

    parameters
        None

    returns 
        None, destructive update
    """
    def set_connective(self):
        for n in self.nx_graph:
            if n.area_label in conn_labels:
                n.set_type("connective")

    """
    converts the graph to json format in a dictionary and 
    saves the dictionary in the specified file location

    parameters
    - location: json file to save to with path, str
        (required)

    returns 
        None
    """
    def to_json(self, location):
        graph_dict = {"nodes": [], "edges": []}

        for n in self.nx_graph:
            (x, y) = n.coordinates
            single = {"id": n.node_id, "name": n.area_label, 
                      "x": int(x), "y": int(y)}
            
            graph_dict["nodes"].append(single)

        edges = self.nx_graph.edges()
        for (n1, n2) in edges:
            single = {"id_1": n1.node_id, "id_2": n2.node_id}
            graph_dict["edges"].append(single)
        
        file = open(location, "w")
        json.dump(graph_dict, file, indent=2)
        file.close()