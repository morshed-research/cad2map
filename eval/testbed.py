import sys
sys.path.append("../")

import os
import json
import numpy as np
from utils.graph import graph, node
from utils.connective import label_list

root = os.path.dirname(__file__) + "/../"
conn_labels = label_list().connective

def make_node(params, scale):
    x = np.int64(params["x"] / scale)
    y = np.int64(params["y"] / scale)
    label = params["name"].lower()

    if "door" in label:
        node_type = "door"
    elif any(s in label for s in conn_labels):
        node_type = "connective"
    else:
        node_type = "area"

    n = node(params["id"], params["name"], 
             (x, y), node_type)
    return n

def make_edge(G, params, node_objs):
    id1 = params["id_1"]
    id2 = params["id_2"]

    n1 = node_objs[id1]
    n2 = node_objs[id2]

    G.nx_graph.add_edge(n1, n2)

def to_graph(json_file, scale=1):
    data = json.load(json_file)
    node_objs = {}

    nodes = data["nodes"]
    edges = data["edges"]
    G = graph()

    for node_data in nodes:
        n = make_node(node_data, scale)

        G.add_node(n)
        node_objs[node_data["id"]] = n

    for edge_data in edges:
        make_edge(G, edge_data, node_objs)

    return G