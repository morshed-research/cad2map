import sys
sys.path.append("../")

from utils.graph import *
from utils.distance import node_dist
from utils.connective import label_list

connective = label_list().connective

"""
creates an edge between every door node and the nearest 
connective area.

parameters
- G: graph to add edges in, graph
    (required)

returns
    graph with new edges, graph
"""
def door_to_connective(G):
    # each type of nodes
    doors = G.get_door_nodes() 
    nodes = G.get_connective_nodes()
    gates = G.get_gate_nodes()
    
    for d in (doors + gates): # go over all door nodes
        # if len(G.get_edges()(d)) > 1:
        #     continue 
        
        distances = {}
        
        # find closest connective node
        for n in nodes:
            dist = node_dist(n, d)[0]
            distances[dist] = n
        
        # add edge to closest
        if len(distances) > 0:
            key = min(distances.keys())
            closest = distances[key]
            G.add_edge(d, closest)
    
    return G

"""
creates an edge between every connective node and the 
nearest n connective nodes

parameters
- G: graph to add edges in, graph
    (required)
- n: number of edges to add per connective node, int
    (default 1)

returns
    graph with new edges, graph
"""
def radial_edges(G, n=1):
    # get connective area nodes
    nodes = [n for n in (G.nx_graph) if n.type == "connective"]

    for n1 in nodes:
        distances = {}
        
        for n2 in nodes: # get distances to all other connective
            if n1 == n2: # no self-edges
                continue
            
            dist = node_dist(n1, n2)[0]
            distances[dist] = n2

        # add edge to smallest n
        for i in range(n):
            if len(distances) == 0: # if less, just fully connected
                break 
            
            key = min(distances.keys())
            closest = distances[key]
            G.add_edge(n1, closest)

            distances.pop(key)

    return G