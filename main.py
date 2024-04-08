from sys import argv
from time import time

t = time()
import pandas as pd

from utils.graph import graph

from slicing.image_slice import slice_images

from extraction.area import get_rooms, make_nodes
from extraction.door import door_boxes, make_doors
from extraction.merge import scale_nodes, merge_dist

from linking.edges import door_edges

from completion.edges import door_to_connective, radial_edges

print(f"imports: {time() - t}")
argc = len(argv)

if __name__ == '__main__':
    if argc != 2 and argc != 3:
        print("usage: python main.py 'image_name'")
        print("alternate usage: python main.py 'image_name' 'slice csv'")
        exit(1)

    image_name = argv[1]

    if argc == 3:
        print(argv[2])
        panels_df = pd.read_csv(argv[2])
    else:
        panels_df = slice_images(image_name, "data/",
                          0.5, 1000,"data/test-panels/")

    G = graph()
    for i in range(len(panels_df)):
        row = panels_df.iloc[i]
        image_path = row["name"]
        print(image_path)

        df = get_rooms(image_path)
        if df.empty:
            continue
        
        local = make_nodes(df)
        door_df = door_boxes(image_path, thresh=0.7)
        local = make_doors(door_df, local)

        G = scale_nodes(local, G, int(row.xmin), int(row.ymin))


    G = merge_dist(G)
    G.set_connective()

    G = door_edges(G)
    G = door_to_connective(G)

    G = radial_edges(G, n=2)

    name = image_name.removesuffix('.jpeg').removesuffix(".png").removesuffix(".jpg")
    G.draw(f"data/{image_name}", 
           f"results/{name}-graph.png", label_it=False)
    
    G.to_json(f"results/json/{name}-graph.json")

    print(f"total run time:{time() - t}secs")