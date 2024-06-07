from bs4 import BeautifulSoup
import sys
import pandas as pd

"""
get_svg: string -> string
REQUIRES: input string is a valid system path to an SVG file
ENSURES: return value is the contents of the SVG pointed to by input path
"""
def get_svg(path):
    file = open(path, "r")
    svg = file.read()

    file.close()
    return svg

"""
get_door_tags: string -> bs4.element.ResultSet
REQUIRES: string is XML of a SVG file
ENSURES: returns a ResultSet of all tags with id = 'Door' inthe given SVG string
"""
def get_door_tags(svg):
    soup = BeautifulSoup(svg, 'xml')
    return soup.find_all(attrs={'id':'Door'})

"""
converts string in format of '(x, y)' into x and y coordinates as integers
"""
def str2coord(s):
    (x, y) = s.split(",")
    return int(float(x)), int(float(y))

"""
get_door_thresh: bs4.element -> string
REQUIRES: given element has id = 'Door' and has a polygon tag with attribute 
          points, depicting a rectangle with coordinate arrangement bottom left, 
          bottom right, top right, top left
ENSURES: returns a string of the first two coordinate pairs of the points 
         attribute, separated by a space
"""
def get_door_thresh(elem):
    polygon = elem.find("polygon")
    coord = polygon.attrs["points"].split(" ")[:-1]
    x = [str2coord(s)[0] for s in coord]
    y = [str2coord(s)[1] for s in coord]

    return x, y

"""
finds the closing coordinate of the bounding box given a polygon element from
the floorplan SVG
"""
def get_curve_end(elem):
    path = elem.find("path")

    coord = path.attrs["d"].split(" ")
    coord = [s.strip(" MmQqLl\n\r\t") for s in coord] # remove other path labels

    x = [str2coord(s)[0] for s in (coord[::2])]
    y = [str2coord(s)[1] for s in (coord[::2])]

    return sum(x), sum(y)

# get all dataset path
if len(sys.argv) != 3:
    sys.exit(1)
else:
    # want to run -> python extract_door_output.py folder path_file
    folder = sys.argv[1]
    paths = open(sys.argv[2], "r")

df=pd.DataFrame(columns=['xmin', 'ymin', 'xmax', 'ymax', 'Frame', 'Label'])

# read each dataset path
for line in paths:
    path = line.strip() # path to output
    svg = get_svg(folder + path + "model.svg")

    door_elements = get_door_tags(svg)

    # write output value for each door
    for elem in door_elements:
        all_x, all_y = get_door_thresh(elem)
        try:
          x, y = get_curve_end(elem)
        except:
          continue

        all_x.append(x)
        all_y.append(y)

        # print(min(all_x), min(all_y), max(all_x), max(all_y))
        df.loc[len(df.index)] = [min(all_x), min(all_y), max(all_x), max(all_y),
                                 path + "F1_scaled.png", "Door"]

print(df)
df.to_csv(folder + "/doors.csv", index=False)
