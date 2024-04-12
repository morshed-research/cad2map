import sys
sys.path.append('../')

from utils.graph import graph, node
from utils.distance import node_dist, dist2pixel
import networkx as nx

def euclidean_weight(n1, n2, edge):
    (dist, centre) = node_dist(n1, n2)
    return dist

def coord_match(n1, n2):
    (x1, y1) = n1.coordinates
    (x2, y2) = n2.coordinates
    one_metre = dist2pixel(1)

    return (abs(x1 - x2) <= one_metre) and (abs(y1 - y2) <= one_metre) 

def find(n, paths):
    for n2 in paths:
        if coord_match(n, n2):
            return n2 
    
    return None

def find_entrance(G):
    for n in G.nx_graph:
        if n.area_label.lower() == "entrance":
            return n 
    
    return n # last node found

class Path_Similarity():
    def __init__(self, G_source, G_target):
        entrance = find_entrance(G_source)
        self.source_paths = nx.shortest_path(G_source.nx_graph, source=entrance, 
                                             weight=euclidean_weight)
        self.source_lengths = nx.shortest_path_length(G_source.nx_graph, source=entrance, 
                                             weight=euclidean_weight)
        
        entrance = find_entrance(G_target)
        self.target_paths = nx.shortest_path(G_target.nx_graph, source=entrance, 
                                             weight=euclidean_weight)
        self.target_lengths = nx.shortest_path_length(G_target.nx_graph, source=entrance, 
                                             weight=euclidean_weight)

    def semantics(self, source_path, target_path):
        matches = 0

        for origin in target_path:
            for n in source_path:
                if (origin.area_label == n.area_label
                    and coord_match(origin, n)):
                    matches += 1
                    break
        
        return matches, len(source_path)
    
    def node_matches(self):
        matches = 0
        total = 0

        for target in self.target_paths:
            source = find(target, self.source_paths)
            if source == None:
                continue

            m, t = self.semantics(self.source_paths[source], 
                                  self.target_paths[target])
            matches += m
            total += t
        
        return matches / total
    
    def weights(self):
        matches = 0
        five = 250

        for target in self.target_lengths:
            source = find(target, self.source_lengths)
            if source == None:
                continue
            
            diff = abs(self.source_lengths[source] - self.target_lengths[target])
            if diff <= five:
                matches += 1

        return matches / len(self.source_lengths)
    
    def evaluate(self):
        nodes = self.node_matches()
        edges = self.weights()

        return (nodes + edges) / 2


