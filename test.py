from eval.testbed import to_graph, to_IL_graph
from eval.metrics import Path_Similarity

import networkx as nx
import gmatch4py as gm

file = open("data/ground-truth/jsons/West-Walkway-graph.json", "r")
target = to_graph(file)
file.close()

file = open("results/json/ground-truth-raw-West-Walkway-labelled-testdoors-graph.json", "r")
source = to_graph(file)
file.close()

target.draw("data/ground-truth/raw/West-Walkway-labelled.png", "results/basic.png")
source.draw("data/ground-truth/raw/West-Walkway-labelled.png", "results/basic-pipe.png")

metric = Path_Similarity(source, target)
print(metric.evaluate())

# attempt for GED
nx1, nx2 = to_IL_graph(target, source)
print(len(nx1.nodes), len(nx2.nodes))
print(len(nx1.edges), len(nx2.edges))
ged = gm.GraphEditDistance(1, 1, 1, 1) # all edit costs are equal to 1
result = ged.compare([nx1, nx2], None) 

print("matrix: ", result)