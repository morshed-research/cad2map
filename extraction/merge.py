import sys
sys.path.append('../')

from utils.graph import *
from utils.distance import *
from utils.connective import label_list

"""
returns the larger string out of the two given strings

parameters
- s1: first string to compare, str
    (required)
- s2: second string to compare, str
    (required)

returns
    string with longer length, str
"""
def select_label(s1, s2):
    if len(s1) > len(s2):
        return s1 
    else:
        return s2
    
"""
merges nodes which represent the same space/object in our graph's map

parameters
- G: graph to operate on, graph
- thresh: euclidean distance threshold to merge based on, int

returns
    updated graph, graph
"""
def merge_thresh_nodes(G, thresh):
    nodes = set(G.nx_graph.nodes) # set of all nodes
    new_G = graph() # new graph with merged nodes

    for n1 in G.nx_graph: # go over all nodes
        if n1 not in nodes: # if removed from set, already merged
            continue
        
        # add nodes that will be merged, you could only be 'merged' to yourself
        rm = set()
        rm.add(n1)
        
        # go over all nodes you could merge to
        for n2 in nodes:
            # if n1 != n2 and n1.type == n2.type == 'door':
            if n1 != n2 and n1.type == n2.type:
                dist, center = node_dist(n1, n2)

                # only merge nodes if euclidean distance under threshold
                if dist <= thresh:
                    n1.coordinates = center
                    new_label = select_label(n1.area_label, n2.area_label)

                    n1_x1, n1_y1 = n1.label_bbox["top_left"]
                    n1_x2, n1_y2 = n1.label_bbox["bottom_right"]
                    n2_x1, n2_y1 = n2.label_bbox["top_left"]
                    n2_x2, n2_y2 = n2.label_bbox["bottom_right"]
                    n1.set_label(new_label)
                    n1.set_bbox(
                        (min(n1_x1, n2_x1), min(n1_y1, n2_y1)),
                        (max(n1_x2, n2_x2), max(n1_y2, n2_y2)),
                    )

                    rm.add(n2) 

        new_G.add_node(n1)

        #remove all merged nodes
        nodes = nodes.difference(rm)

    # update id accordingly, so new nodes added after even merged ones
    new_G.next_id = G.next_id
    return new_G

"""
merges all nodes that are within threshold metres in the given graph

parameters
- G: graph to merge nodes in, graph
    (required)

returns 
    graph with nodes merged, graph
"""
def merge_by_dist(G):
    threshold = dist2pixel(0.5)
    return merge_thresh_nodes(G, threshold)

"""
merges nodes if their labels have semantic meaning when combined

parameters
- G: graph to merge nodes in, graph
    (required)

returns 
    graph with nodes merged, graph
"""
def merge_labels(G):
    nodes = set(G.nx_graph.nodes) # set of all nodes
    new_G = graph() # new graph with merged nodes

    for n1 in G.nx_graph: # go over all nodes
        if n1 not in nodes: # if removed from set, already merged
            continue
        
        # add nodes that will be merged, you could only be 'merged' to yourself
        rm = set()
        rm.add(n1)
        
        # go over all nodes you could merge to
        for n2 in nodes:
            if n1 != n2 and n1.type == n2.type:
                dist, (x, y) = node_dist(n1, n2)

                # only merge nodes if combined label in connective list
                new_label = n1.area_label + n2.area_label
                if dist <= 100 and (new_label in label_list().all or n2.area_label == "room"):
                    n1.set_coordinates(x, y)

                    n1_x1, n1_y1 = n1.label_bbox["top_left"]
                    n1_x2, n1_y2 = n1.label_bbox["bottom_right"]
                    n2_x1, n2_y1 = n2.label_bbox["top_left"]
                    n2_x2, n2_y2 = n2.label_bbox["bottom_right"]
                    n1.set_label(new_label)
                    n1.set_bbox(
                        (min(n1_x1, n2_x1), min(n1_y1, n2_y1)),
                        (max(n1_x2, n2_x2), max(n1_y2, n2_y2)),
                    )

                    rm.add(n2) 

        new_G.add_node(n1)

        #remove all merged nodes
        nodes = nodes.difference(rm)

    # update id accordingly, so new nodes added after even merged ones
    new_G.next_id = G.next_id
    return new_G

"""
Takes the nodes in G, scales their coordinates by x + xmin, y + ymin
and then adds each of these into the given global graph

parameters:

"""
def scale_nodes(G, global_graph, xmin, ymin):
    for n in G.nx_graph:
        (x, y) = n.coordinates
        x1, y1 = n.label_bbox["top_left"]
        x2, y2 = n.label_bbox["bottom_right"]

        n.set_coordinates(x + xmin, y + ymin)
        n.set_label(n.area_label) 
        n.set_bbox((x1+xmin, y1+ymin), (x2+xmin, y2+ymin))
        n.node_id = global_graph.next_id

        global_graph.add_node(n)

    return global_graph