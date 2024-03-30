from utils.graph import *

"""
creates door nodes based on given door bounding boxes and adds them to 
the given graph

parameters
- door_df: dataframe with door bounding boxes with 
           columns 'xmin' 'ymin' 'xmax' 'ymax', pd.DataFrame
    (required)
- G: graph to add nodes to, graph
    (required)

returns 
    graph with door nodes added, graph
"""
def make_doors(door_df, G):
    for i, row in door_df.iterrows():
        x1, y1, x2, y2 = row["xmin"], row["ymin"], row["xmax"], row["ymax"]

        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        G.add_node(node(id=G.next_id, area="Door", coord=center, type="door"))

    return G