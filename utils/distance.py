from .graph import *
import math

"""
converts pixels to actual distance using assumed ratio

parameters
- pixels: number of pixels to convert, int
    (required)

returns
    conversion value, int
"""
def pixel2dist(pixels):
    return pixels

"""
converts actual distance to pixel equivalent by assumed ratio

parameters
- distance: distance to convert, int
    (required)

returns
    conversion value, int
"""
def dist2pixel(distance):
    return distance * 50

"""
calculates the euclidean distance and center coordinate between
two nodes

parameters
- n1: first node, node
    (required)
- n2: second node, node
    (required)

returns 
    distance, int * center, (int * int)
"""
def node_dist(n1, n2):
    (x1, y1) = n1.coordinates
    (x2, y2) = n2.coordinates

    # calculating each property
    dist = int(math.sqrt((x1 - x2)**2 + (y1 - y2)**2))
    center = ((x1 + x2) // 2, (y1 + y2) // 2)

    return dist, center

"""
returns the euclidean distance between two nodes to use as the
edge weight between them

parameters:
- n1: first node on edge, node
    (required)
- n2: second node on edge, node
    (required)
- edge: edge properties, dict
    (required, unused)

returns
    euclidean distance, int
"""
def euclidean_weight(n1, n2, edge):
    (dist, centre) = node_dist(n1, n2)
    return dist

"""
returns true if nodes are within 1 metre of each other, false otherwise 

parameters:
- n1: first node to check, node
    (required)
- n2: second node to compare to, node
    (required)

returns
    bool
"""
def coord_match(n1, n2):
    (x1, y1) = n1.coordinates
    (x2, y2) = n2.coordinates
    metre = dist2pixel(2)

    return (abs(x1 - x2) <= metre) and (abs(y1 - y2) <= metre) 

"""
finds first node in paths that is within 1 metre of node n

parameters:
- n: node to find, node
    (required)
- paths: nodes to search amongst, list or dict
    (required)

returns
    node close to n, node
"""
def find(n, paths):
    if n == None:
        return None
    
    final = None
    min_dist = None

    for n2 in paths:
        (dist, center) = node_dist(n, n2)
        if min_dist == None:
            min_dist = dist 

        if n == n2 and dist <= min_dist:
            final = n2
    
    return final

"""
returns node with label entrance, else None

parameters:
- G: graph to search, graph
    (required)

returns 
    node
"""
def find_entrance(G):
    for n in G.nx_graph:
        if "entrance" in n.area_label.lower():
            return n
    
    return None

"""
returns appropriate start points for paths between two graphs

parameters
- G1: first graph to evaluate, graph
    (required)
- G2: second graph to evaluate, graph
    (required)

returns 
    node * node
"""
def find_match(G1, G2):
    n1 = find_entrance(G1)
    n2 = find(n1, G2.nx_graph)

    if n1 == None or n2 == None:
        for n1 in G1.nx_graph:
            n2 = find(n1, G2.nx_graph)

            if n2 != None:
                return n1, n2
            
    return n1, n2 