import sys
sys.path.append("../")

import os
import json
import numpy as np
from utils.graph import graph, node
from utils.connective import label_list

root = os.path.dirname(__file__) + "/../"
conn_labels = label_list().connective

"""
creates a node from the JSON file structure information

parameters
- params: dictionary from JSON structure of single node. 
          Specified with id name x y, dict
    (required)
- scale: scale to add coordinates with, float
    (required)

retuns
    node object with given specs, node
"""
def make_node(params: dict, scale: float):
    x = np.int64(params["x"] / scale)
    y = np.int64(params["y"] / scale)
    label = params["name"].lower()

    # get type
    if "door" in label:
        node_type = "door"
    elif any(s in label for s in conn_labels):
        node_type = "connective"
    else:
        node_type = "area"

    # final node
    n = node(params["id"], params["name"], 
             (x, y), node_type)
    return n


"""
destructively adds an edge to graph 
from the JSON file structure information

parameters
- G: graph to add to, graph
- params: dictionary from JSON structure of single node. 
          Specified with id_1 id_2, dict
    (required)
- node_objs: dictionary containing all nodes of the graph as values, 
             keyed by node id, dict
    (required)

retuns
    None
"""
def make_edge(G, params, node_objs):
    id1 = params["id_1"]
    id2 = params["id_2"]

    n1 = node_objs[id1]
    n2 = node_objs[id2]

    G.nx_graph.add_edge(n1, n2)

"""
converts the JSON file at the given path into 
a graph object, with coordinates scaled at the given 
scale 

parameters
- json_file: path of JSON file, str
    (required)
- scale: scale for coordinates, float
    (default = 1)

returns
    graph from JSON specs, graph
"""
def to_graph(json_file, scale=1):
    data = json.load(json_file)
    node_objs = {}

    nodes = data["nodes"]
    edges = data["edges"]
    G = graph()

    for node_data in nodes:
        n = make_node(node_data, scale)
        n.test = True # for evaluation, so testing mode equivalence

        G.add_node(n)
        node_objs[node_data["id"]] = n

    for edge_data in edges:
        make_edge(G, edge_data, node_objs)

    return G