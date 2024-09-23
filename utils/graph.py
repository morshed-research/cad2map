import sys
sys.path.append('../')

import cv2
import json
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from .connective import label_list
from .distance import coord_match

node_area_types = ["area", "door", "connective", "gate"]
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
    def __init__(self, id: int, area: str="N/A", label_bbox: dict={}, coord=(-1,1), type: str="area"):
        self.node_id = id # id in graph
        self.area_label = area # label from annotation
        self.label_bbox = label_bbox # bounding box of the label
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

    def set_bbox(self, top_left: tuple, bottom_right: tuple):
        self.label_bbox = {"top_left":top_left, "bottom_right": bottom_right}

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
    get the coordinates of node

    returns
        (x,y) corrdinates of the nodes
    """
    def get_coordinates(self):
        return self.coordinates

    """
    get node type
    """
    def get_type(self):
        return self.type

    """
    get node bounding box
    """
    def get_bbox(self):
        return self.label_bbox

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
    Add edge to the graph
    """
    def add_edge(self, n1, n2):
        self.nx_graph.add_edge(n1, n2)

    def _draw_segmentation(self, image, segments: list, segmentation_type='fill',
                           fill_color=[0,0,255]):
        # Loop through thesegments and fill them
        for area in tqdm(segments, "Filling Segments"):
            if segmentation_type=='fill':
                for (x,y) in segments[area]:
                    image[y,x] = fill_color
            elif segmentation_type=='box':
                x1, y1 = segments[area]['top_left']
                x2, y2 = segments[area]['bottom_right']
                image[y1:y2+1, x1:x2+1] = fill_color

        return image


    def _draw_graph(self, image, node_color, edge_color, area_bbox_color, door_bbox_color):
        nodes = self.get_nodes()
        edges = self.get_edges()

        for node in nodes:
            # draw node
            x,y = node.get_coordinates()
            cv2.circle(image, (x,y), 3, node_color, 1)
            image[y, x] = node_color

            # draw bbox
            node_type = node.get_type()
            node_bbox = node.get_bbox()
            if node_type == "door" or node_bbox == "area":
                x1, y1 = node_bbox['top_left']
                x2, y2 = node_bbox['bottom_right']
                if node_type == "area":
                    cv2.rectangle(image, (x1, y1), (x2, y2), area_bbox_color, 5)
                else:
                    cv2.rectangle(image, (x1, y1), (x2, y2), door_bbox_color, 5)
        
        # draw edges
        for (n1, n2) in edges:
            x1, y1 = n1.get_coordinates()
            x2, y2 = n2.get_coordinates()
            cv2.line(image, (x1, y1), (x2, y2), edge_color, 3)

        return image


    """
    draws the graph on top of the input image

    parameters
    - image_path: location of input image, str (required)
    - area_segments: all segments of areas, list[set] (required)
    - hallway_segments: all segments of hallways, list[set] (required)
    - display: displays the image at the end of the function, bool (default False)
    - save_path: where to save resulting plot, str (required)

    returns
        None
    """
    def draw(self, image_path, area_segments, hallway_segments, display=False, save_path=''):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        # draw area and hallway segments
        image = self._draw_segmentation(image, area_segments, 'fill', [0,0,255])
        image = self._draw_segmentation(image, {'hallway': hallway_segments}, 'fill', [255,0,0])

        # draw graph
        image = self._draw_graph(image, [255, 255, 0], [0, 255, 255], 
                                        [0, 255, 0], [255, 192, 203])

        if save_path != '':
            cv2.imwrite(save_path, image)

        if display:
            cv2.imshow('Result', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


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

        edges = self.get_edges()()
        for (n1, n2) in edges:
            single = {"id_1": n1.node_id, "id_2": n2.node_id}
            graph_dict["edges"].append(single)
        
        file = open(location, "w")
        json.dump(graph_dict, file, indent=2)
        file.close()

    """
    Get nodes of the graph     
    """
    def get_nodes(self):
        return self.nx_graph.nodes
    
    """
    Get edges
    """
    def get_edges(self):
        return self.nx_graph.edges

    """
    Get bounding boxes of all door nodes     
    """
    def get_door_bboxes(self):
        nodes_list = list(self.get_nodes())
        bboxes = []
        for node in nodes_list:
            if node.type == "door":
                bboxes.append(node.label_bbox)
        return bboxes

    """
    Get bounding boxes of all area nodes
    """
    def get_area_bboxes(self):
        nodes_list = list(self.get_nodes())
        bboxes = []
        for node in nodes_list:
            if node.type == "area":
                bboxes.append(node.label_bbox)
        return bboxes
    
    """
    Get all door nodes
    """
    def get_door_nodes(self):
        nodes_list = list(self.get_nodes())
        nodes = []
        for node in nodes_list:
            if node.type == "door":
                nodes.append(node)
        return nodes
    
    
    """
    Get all area nodes
    """
    def get_area_nodes(self):
        nodes_list = list(self.get_nodes())
        nodes = []
        for node in nodes_list:
            if node.type == "area":
                nodes.append(node)
        return nodes
    
    """
    Get all connective nodes
    """
    def get_connective_nodes(self):
        nodes_list = list(self.get_nodes())
        nodes = []
        for node in nodes_list:
            if node.type == "connective":
                nodes.append(node)
        return nodes
    
    """
    Get all gate nodes
    """
    def get_gate_nodes(self):
        nodes_list = list(self.get_nodes())
        nodes = []
        for node in nodes_list:
            if node.type == "gate":
                nodes.append(node)
        return nodes
    
