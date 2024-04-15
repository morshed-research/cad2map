from eval.testbed import to_graph
from eval.metrics import Path_Similarity

import networkx as nx
import gmatch4py as gm

file = open("data/ground-truth/jsons/test_floorplan.json", "r")
target = to_graph(file)
file.close()

file = open("results/json/test-floorplan-labelled-graph.json", "r")
source = to_graph(file)
file.close()

target.draw("data/test-floorplan-labelled.png", "results/basic.png")
source.draw("data/test-floorplan-labelled.png", "results/basic-pipe.png")

metric = Path_Similarity(source, target)
print(metric.evaluate())

ged = gm.GraphEditDistance(1, 1, 1, 1) # all edit costs are equal to 1
result = ged.compare([source.nx_graph,target.nx_graph], None) 

print("matrix: ", result)
print("similarity: ", ged.similarity(result))