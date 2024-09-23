import sys
sys.path.append('../')

from utils.distance import dist2pixel, euclidean_weight, find, find_match
import networkx as nx


"""
Class that implements our semantic path similarity evaluation metric

properties are not needed for class use, create per eval pair

create instance
- Path_Similarity(generated_graph, gt_graph)

run evaluation
- object.evaluate()
"""
class Path_Similarity():

    """
    create metric instance

    parameters
    - G_source: generated graph to evaluate, graph
        (required)
    - G_target: ground truth graph to evaluate against, graph
        (required)

    returns
        Path_Similarity object
    """
    def __init__(self, G_source, G_target):
        # start points in each graph
        s1, s2 = find_match(G_target, G_source)
        if s1 == None or s2 == None:
            print(0, 0, 0, 0)
            exit(0)

        # all target paths - dijkstra
        self.target_paths = nx.shortest_path(G_target.nx_graph, source=s1, 
                                             weight=euclidean_weight)
        self.target_lengths = nx.shortest_path_length(G_target.nx_graph, source=s1, 
                                             weight=euclidean_weight)
        
        # all source paths - dijkstra
        self.source_paths = nx.shortest_path(G_source.nx_graph, source=s2, 
                                             weight=euclidean_weight)
        self.source_lengths = nx.shortest_path_length(G_source.nx_graph, source=s2, 
                                             weight=euclidean_weight)
        
        # print testbed stats
        G1_nodes = list(G_target.get_nodes())
        doors = [n for n in G1_nodes if n.type == "door"]
        nodes = [n for n in G1_nodes if n.type == "area"]

        paths = self.target_lengths.values()

        print(f"Area Nodes\t Door Nodes\t Total Paths\t Longest Path\t Shortest Path\t Average Path")
        print(f"{len(nodes)}\t\t {len(doors)}\t\t {len(paths)}\t\t {max(paths)}\t\t {min(paths)}\t\t {sum(paths) / len(paths)}")

    """
    private function to check whether the nodes on
    the target path are present on the source path

    parameters
    - source_path: path to evaluate, dict with nodes as keys or list of nodes
    - target_path: path to evaluate against, dict with nodes as keys or list of nodes

    returns 
        proportion of missing nodes in path, int
    """
    def __semantics(self, source_path, target_path):
        matches = 0

        for origin in target_path:
            for n in source_path:

                if origin == n: #found match
                    matches += 1
                    break # resume to next target node
        
        return len(target_path) - matches
    
    """
    private function to average number of missing nodes per path,
    for area nodes only

    parameters
        None

    returns 
        average number of missing nodes in path, int
    """
    def __semantic_matches(self):
        matches = []
        total = 0

        for target in self.target_paths:
            if target.type != "area": # only for destinations
                continue 

            source = find(target, self.source_paths)
            if source == None: # no equivalent node 
                continue
            
            # another path evaluated
            total += 1

            # update number of missing nodes
            path_len =  len(self.target_paths[target])
            diff = self.__semantics(self.source_paths[source], self.target_paths[target])

            matches.append(diff / path_len)

        if total == 0:
            return 0
        else:
            return sum(matches) / total
    
    """
    private function to check how many target nodes are
    present in source graph

    parameters:
        None

    returns 
        ratio of matching nodes over total nodes, float
    """
    def __node_matches(self, type):
        matches = 0
        total = 0

        for target in self.target_paths:
            if target.type != type:
                continue 
            
            total += 1 # searching for another node 

            source = find(target, self.source_paths)
            if source == None: # no equivalent node
                continue

            matches += 1
        
        if total == 0:
            return 0
        else:
            return matches / total
    
    """
    private function to check how many paths to area nodes have distance
    within 5m of target path

    parameters
    - verbose: print failed paths, bool
        (default = False)

    return  
        proportion of passing paths compared to total evaluated paths
    """
    def __weights(self, verbose=False):
        matches = []
        total = 0

        for target in self.target_lengths:
            if target.type != "area": # only paths to destinations
                continue 

            source = find(target, self.source_lengths)
            if source == None: # no equivalent node, already penalised in node matching
                continue
            
            # another path evaluated
            total += 1
            diff = abs(self.source_lengths[source] - self.target_lengths[target])
            
            if self.target_lengths[target] == 0:
                matches.append(0)
            elif (diff / self.target_lengths[target]) <= 0.2: # path with up to 20% difference
                matches.append(diff / self.target_lengths[target])
            elif verbose:
                # show missing path if requested
                print("\n\ndiff:", diff / self.target_lengths[target], "s:", source, "t:", target, 
                      "\nsource len:", self.source_paths[source], self.source_lengths[source], 
                      "\ntarget len:", self.target_paths[target], self.target_lengths[target])

        if total == 0:
            return 0
        else:
            return sum(matches) / total
    
    """
    runs the evaluation metric. 

    parameters
        None
    
    returns
        ratio of matching nodes * ratio of matching path weights * average missing nodes per path
    """
    def evaluate(self):
        door_nodes = self.__node_matches("door")
        area_nodes = self.__node_matches("area")
        edges = self.__weights()
        semantics = self.__semantic_matches()

        print("Door Node Match, Area Node Match, Weighted Path Similarity, Semantic Path Similarity")
        return door_nodes, area_nodes, edges, semantics


