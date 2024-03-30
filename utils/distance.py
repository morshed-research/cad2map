from .graph import *
import math

"""
converts pixels to actual distance using assumed ratio

parameters
- pixels: number of pixels to convert, int
    (required)

returns
    conversion value, int
"""
def pixel2dist(pixels):
    return pixels

"""
converts actual distance to pixel equivalent by assumed ratio

parameters
- distance: distance to convert, int
    (required)

returns
    conversion value, int
"""
def dist2pixel(distance):
    return 100

"""
calculates the euclidean distance and center coordinate between
two nodes

parameters
- n1: first node, node
    (required)
- n2: second node, node
    (required)

returns 
    distance, int * center, (int * int)
"""
def node_dist(n1, n2):
    (x1, y1) = n1.coordinates
    (x2, y2) = n2.coordinates

    # calculating each property
    dist = int(math.sqrt((x1 - x2)**2 + (y1 - y2)**2))
    center = ((x1 + x2) // 2, (y1 + y2) // 2)

    return dist, center