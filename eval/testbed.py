import sys
sys.path.append("../")

import os
import json
import numpy as np
import networkx as nx 

from utils.graph import graph, node
from utils.connective import label_list
from utils.distance import find

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

"""
extracts only door nodes from the given JSON file and adds it to the given graph

parameters
- G: graph to add door nodes to, graph
    (required)
- json_file: path of JSON file, str
    (required) 
- scale: scale for coordinates, float
    (default = 1)

returns
    graph with doors from JSON file, graph
"""
def test_doors(G, json_file, scale=1):
    data = json.load(json_file) # load file data

    nodes = data["nodes"] # read all nodes
    for node_data in nodes:
        n = make_node(node_data, scale)

        if n.type == "door": # only add door nodes
            n.node_id = G.next_id
            G.add_node(n)
            print("door added")

    return G

"""
Removes the given node from the list of nodes

parameters
- nodes: list of nodes to remove from, node list
    (required)
- rn: node to remove, node
    (required) 

returns 
    None
"""
def remove_node(nodes, rn):
    for i in range(len(nodes)):
        n = nodes[i]
        if n.node_id == rn.node_id: # found match by id
            nodes.pop(i)
            return
        
    print("not found")

"""
converts the given two graphs to integer labelled graphs, with matching nodes within the graphs having the same integer id

parameters
- G1: first graph to convert, graph
- G2: second graph to convert, graph

returns
    first IL graph & second IL graph, graph * graph
"""
def to_IL_graph(G1, G2):
    # store integer mappings for edges
    G1_map = dict()
    G2_map = dict()

    # list of each's nodes
    G1_nodes = list(G1.nx_graph.nodes)
    G2_nodes = list(G2.nx_graph.nodes)

    IL_1 = nx.Graph()
    IL_2 = nx.Graph()

    id = 0
    missing = 0

    for n1 in G1_nodes: # process first graph's nodes
        n2 = find(n1, G2_nodes)

        G1_map[n1] = id # update mapping
        IL_1.add_node(id)

        if n2 != None: # make equivalent node have same id
            G2_map[n2] = id
            IL_2.add_node(id)

            remove_node(G2_nodes, n2) # don't double process later
        else:
            missing += 1 # record number of no matches

        id += 1

    for n2 in G2_nodes: # process all remaining in second
        G2_map[n2] = id
        IL_2.add_node(id)

        id += 1 # continue same id

    # process edges based on new labels
    for (n1, n2) in G1.nx_graph.edges:
        IL_1.add_edge(G1_map[n1], G1_map[n2])

    for (n1, n2) in G2.nx_graph.edges:
        IL_2.add_edge(G2_map[n1], G2_map[n2])

    return IL_1, IL_2
    
