import sys
sys.path.append("../")

from utils.graph import *
from utils.distance import node_dist

"""
connects every non-door node to the closest door node in
the given graph

parameters
- G: graph to add edges in, graph
    (required)

returns
    graph with new edges, graph
"""
def door_edges(G):
    all_nodes = list(G.nx_graph.nodes())
    doors = [n for n in all_nodes if n.type == "door"]
    nodes = [n for n in all_nodes if n.type != "door"]

    for n in nodes:
        distances = {}
        
        for d in doors:
            dist = node_dist(n, d)[0]
            distances[dist] = d

        key = min(distances.keys())
        closest = distances[key]
        G.nx_graph.add_edge(n, closest)

    return G