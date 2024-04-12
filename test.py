from eval.testbed import to_graph
from eval.metrics import Path_Similarity

file = open("data/ground-truth/jsons/test_floorplan.json", "r")
target = to_graph(file)
file.close()

file = open("results/json/test-floorplan-labelled-graph.json", "r")
source = to_graph(file)
file.close()

target.draw("data/test-floorplan-labelled.png", "results/basic.png")
source.draw("data/test-floorplan-labelled.png", "results/basi-pipe.png")

metric = Path_Similarity(source, target)
print(metric.evaluate())

# G.draw("data/ground-truth/raw/West-Walkway.png", "results/west-walkway-testbed.png")