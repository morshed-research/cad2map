from eval.testbed import to_graph

file = open("data/ground-truth/jsons/West-Walkway-graph.json", "r")
G = to_graph(file, 0.42)

G.draw("data/ground-truth/raw/West-Walkway.png", "results/west-walkway-testbed.png")