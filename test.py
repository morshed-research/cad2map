from eval.testbed import to_graph, to_IL_graph
from eval.metrics import Path_Similarity

import networkx as nx
import gmatch4py as gm

from sys import argv

# get command line arguments
argc = len(argv)
if argc != 4:
    print("usage: python main.py <path-to-image-json> <path-to-ground-truth-json> <path-to-original-image>")
else:
    source_json = argv[1]
    target_json = argv[2]
    image_path = argv[3]

# target graph
file = open(target_json, "r")
target = to_graph(file)
file.close()

# generated graph
file = open(source_json, "r")
source = to_graph(file)
file.close()

# show graphs as images
target.draw(image_path, "results/basic.png")
source.draw(image_path, "results/basic-pipe.png")

# path-based similarity
metric = Path_Similarity(source, target)
print(metric.evaluate())

# attempt for GED
nx1, nx2 = to_IL_graph(target, source)
print(len(nx1.nodes), len(nx2.nodes))
print(len(nx1.edges), len(nx2.edges))
ged = gm.GraphEditDistance(1, 1, 1, 1) # all edit costs are equal to 1
result = ged.compare([nx1, nx2], None) 

print("matrix: ", result)