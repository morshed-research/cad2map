import sys
sys.path.append('../')

from utils.graph import graph, node
from utils.distance import node_dist, dist2pixel, euclidean_weight, find, find_entrance, coord_match
import networkx as nx

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
                if origin == n:
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


