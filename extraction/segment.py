import sys
sys.path.append('../')

import cv2
import numpy as np
from tqdm import tqdm
from utils.graph import node


def image_preprocessing(image_path, morph_kernal_size=(3,3)):
    """
    Uploads and preprocesses the image to ensure black lines are connected.
    
    Args:
        image_path (str): Path to the input image.
        morph_kernal_size TUPLE(int, int): Kernel size for morphological operations.
        
    Returns:
        processed_image (np.ndarray):
            The preprocessed binary image with connected lines. To access (y,x).
    """
    # Load the CAD drawing, threshold it, and invert it
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Perform morphological openning to fill gaps in lines
    kernel = np.ones(morph_kernal_size, np.uint8)
    processed_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

    return processed_image


def touches_black_line(image, bounding_box, direction):
    x1, y1 = bounding_box['top_left']
    x2, y2 = bounding_box['bottom_right']
    
    if direction != "h" and direction != "v":
        raise Exception("Wrong Direction")
    
    if direction == "v":
        # Check top edge
        if np.any(image[y1, x1:x2+1] == 0):
            return True
        # Check bottom edge
        if np.any(image[y2, x1:x2+1] == 0):
            return True
    elif direction == "h":
        # Check left edge
        if np.any(image[y1:y2+1, x1] == 0):
            return True
        # Check right edge
        if np.any(image[y1:y2+1, x2] == 0):
            return True
    
    return False

# Expand the bounding box horizontally until it touches a black line
def expand_bounding_box_horizontally(image, bounding_box):
    x1, y1 = bounding_box['top_left']
    x2, y2 = bounding_box['bottom_right']
    
    # Expand the bounding box horizontally
    x1 = max(0, x1 - 1)
    x2 = min(image.shape[1] - 1, x2 + 1)
    new_bounding_box = {"top_left": (x1, y1), "bottom_right": (x2, y2)}
    
    while not touches_black_line(image, new_bounding_box, "h"):
        # Expand the bounding box horizontally
        x1 = max(0, x1 - 1)
        x2 = min(image.shape[1] - 1, x2 + 1)
        new_bounding_box = {"top_left": (x1, y1), "bottom_right": (x2, y2)}
    
    final_bounding_box = {"top_left": (x1-1, y1), "bottom_right": (x2-1, y2)}
    return final_bounding_box

# Expand the bounding box vertically until it touches a black line
def expand_bounding_box_vertically(image, bounding_box):
    x1, y1 = bounding_box['top_left']
    x2, y2 = bounding_box['bottom_right']
    
    # Expand the bounding box vertically
    y1 = max(0, y1 - 1)
    y2 = min(image.shape[0] - 1, y2 + 1)
    new_bounding_box = {"top_left": (x1, y1), "bottom_right": (x2, y2)}
    
    while not touches_black_line(image, new_bounding_box, "v"):
        # Expand the bounding box vertically
        y1 = max(0, y1 - 1)
        y2 = min(image.shape[0] - 1, y2 + 1) 
        new_bounding_box = {"top_left": (x1, y1), "bottom_right": (x2, y2)}
    
    final_bounding_box = {"top_left": (x1, y1-1), "bottom_right": (x2, y2-1)}
    return final_bounding_box


def flood_fill(image, start_point):
    h, w = image.shape
    filled_coords = set()
    stack = [start_point]

    while stack:
        x, y = stack.pop()
        
        if 0 <= x < w and 0 <= y < h:
            if image[y, x] == 255 and (x, y) not in filled_coords:
                filled_coords.add((x, y))
                stack.append((x - 1, y))
                stack.append((x + 1, y))
                stack.append((x, y - 1))
                stack.append((x, y + 1))

    return filled_coords


def segment_areas_box(image_path, bounding_boxes):
    """
    For each bounding box which is within a room, segment the room by calculating
    the biggest box you can fit in the room.

    Args:
        image_path (str): Path to the image we are segmenting.
        bounding_boxes ([{"top_left":(int,int), "bottom_right":(int,int)}]):
            Bounding boxes of all the centers of the room to be segmented.

    Returns:
        [[{"top_left":(int,int), "bottom_right":(int,int)}]]: 
            Bounding boxxes of all the segments.
    """
    # Load the CAD drawing
    image = image_preprocessing(image_path)

    # Loop through the boundingboxes and segment them
    segments = []
    for bounding_box in tqdm(bounding_boxes, "Segmenting Areas"):
        bounding_box_h = expand_bounding_box_horizontally(image, bounding_box)
        bounding_box_v = expand_bounding_box_vertically(image, bounding_box_h)
        segments.append(bounding_box_v)

    return segments


def segment_areas_fill(G, image_path):
    """
    For each bounding box which is within a room, segment the room by flood fill.

    Args:
        image_path (str): Path to the image we are segmenting.
        bounding_boxes ([{"top_left":(int,int), "bottom_right":(int,int)}]):
            Bounding boxes of all the centers of the room to be segmented.

    Returns:
        {area_node:{(int,int), ...}, ...}: 
            Coordinates all the segments.
    """
    # Load the CAD drawing
    image = image_preprocessing(image_path)
    nodes_list = G.get_area_nodes()

    # Loop through the boundingboxes and segment them
    segments = {}

    # for bounding_box in tqdm(bounding_boxes, "Segmenting Areas"):
    for node in tqdm(nodes_list, "Segmenting Areas"):
        bbox = node.get_bbox()

        tl_filling = flood_fill(image, bbox['top_left'])
        br_filling = set()
        if bbox['bottom_right'] not in tl_filling: 
            br_filling = flood_fill(image, bbox['bottom_right'])
        filled_room_coords = tl_filling | br_filling
        segments[node] = filled_room_coords

    return segments


def is_segmented(point, segmented_coords):
    for area in segmented_coords:
        if point in segmented_coords[area]:
            return True, area
    return False, None


def is_hallway_door(image, door_bbox, segmented_areas):
    """
    A hallway door is a door that intersects an already segmented "area" and
    a non-segmented area which is assumed to be a hallway.
    
    Args:
        image (np.ndarray): The preprocessed binary image.
        door_bbox (tuple): Bounding box of the door in the format (x1, y1, x2, y2).
        segmented_areas (list): List of lists where each list contains the (y, x)
          coordinates of a segmented area.
    
    Returns:
        bool: True if the door is a hallway door, False otherwise.
    """
    x1, y1 = door_bbox['top_left']
    x2, y2 = door_bbox['bottom_right']
    
    touches_segmented = False
    touches_non_segmented = False

    for y in range(y1, y2):
        for x in range(x1, x2):
            if image[y, x] == 255:
                is_seg, _ = is_segmented((x, y), segmented_areas)
                if is_seg:
                    touches_segmented = True
                else:
                    touches_non_segmented = True

            if touches_segmented and touches_non_segmented:
                return True
            
    return False


def is_connective_door(image, door_bbox, segmented_hallways):
    """
    A connective door is a door that fully occurs within a segmented hallway.
    
    Args:
        image (np.ndarray): The preprocessed binary image.
        door_bbox (tuple): Bounding box of the door in the format (x1, y1, x2, y2).
        segmented_areas (list): List of lists where each list contains the (y, x)
          coordinates of a segmented area.
    
    Returns:
        bool: True if the door is a hallway door, False otherwise.
    """
    x1, y1 = door_bbox['top_left']
    x2, y2 = door_bbox['bottom_right']
    
    door_coords = set()

    for y in range(y1, y2):
        for x in range(x1, x2):
            if image[y, x] == 255:
                is_seg, _ = is_segmented((x, y), segmented_hallways)
                if not is_seg:
                    return False, set()
            door_coords.add((x, y))
            
    return True, door_coords


def find_hallway_filling_start_point(image, door_bbox, segmented_areas):
    x1, y1 = door_bbox['top_left']
    x2, y2 = door_bbox['bottom_right']
    
    start_point, node = None, None
    for y in range(y1, y2):
        for x in range(x1, x2):
            is_seg, node = is_segmented((x, y), segmented_areas)
            
            if image[y, x] == 255 and not is_seg:
                start_point = (x, y)

            if start_point and node:
                return start_point, node
    
    return start_point, node


def segment_hallways_fill(G, image_path, segmented_areas):
    """
    Starting from doors, start to segment the hallways, and connect the doors
    to their room. In addition, remove doors that connect two connective areas.
    Finally, create gate doors.

    Args:
        G: The graph we built so far.
        image_path (str): Path to the image we are segmenting.
        segmented_areas:

    Returns:
        G, {(int,int), ...}: Graph, Coordinates of all the segments.
    """
     # Load the CAD drawing
    image = image_preprocessing(image_path)
    nodes_list = list(G.get_door_nodes())
    # bboxes_list = list(G.get_door_bboxes())

    hallways = set()
    connective_entrance_doors = []
    # for bbox in tqdmc(bboxes_list, "Segmenting Hallways"):
    for door_node in tqdm(nodes_list, "Segmenting Hallways"):
        bbox = door_node.get_bbox()

        if is_hallway_door(image, bbox, segmented_areas):
            start_point, area_node = find_hallway_filling_start_point(image, bbox, segmented_areas)
            
            if start_point:
                if start_point not in hallways:
                    hallway_coords = flood_fill(image, start_point)
                    hallways.update(hallway_coords)
                G.add_edge(area_node, door_node)
        else:
            connective_entrance_doors.append(door_node)

    # Find connective doors and add their pixels to hallways and create gates
    hallways_dict = {'hallways': hallways}
    for door_node in tqdm(connective_entrance_doors, "Segmenting Connective doors"):
        bbox = door_node.get_bbox()

        is_con, coords = is_connective_door(image, bbox, hallways_dict)
        if is_con:
            hallways.update(coords)
            door_node.set_type("connective")

        else:
            door_node.set_type("gate")
    
    return G, hallways


def reduce_pixel_density(pixels, width, height, row_factor, col_factor, wall_distance=10):
    """
    Reduce the density of a set of pixels by merging rows and columns.
    
    Args:
        pixels (set of tuples): Set of (x, y) coordinates of the pixels.
        width (int): Width of the grid.
        height (int): Height of the grid.
        row_factor (int): Factor by which to reduce the number of rows.
        col_factor (int): Factor by which to reduce the number of columns.
    
    Returns:
        dict: Keys are centroids (x, y) and values are sets of original pixels.
    """
    centroids = {}

    # Create an empty grid of the given width and height and mark pixels
    grid = np.zeros((height, width), dtype=bool)
    for x, y in pixels:
        grid[y, x] = True

    # Calculate the new grid dimensions
    new_height = height // row_factor
    new_width = width // col_factor

    # Helper function to check if a centroid is near a wall
    def is_near_wall(centroid_x, centroid_y, wall_distance):
        for dy in range(-wall_distance, wall_distance + 1):
            for dx in range(-wall_distance, wall_distance + 1):
                nx, ny = centroid_x + dx, centroid_y + dy
                if (nx, ny) not in pixels and (0 <= nx < width) and (0 <= ny < height):
                    if not grid[ny, nx]:  # If it's a wall pixel
                        return True
        return False

    # Traverse the grid in blocks
    for i in tqdm(range(new_height), "Reducing Pixels Density in Hallways"):
        for j in range(new_width):
            # Determine the bounds of the current block
            y_start = i * row_factor
            y_end = (i + 1) * row_factor
            x_start = j * col_factor
            x_end = (j + 1) * col_factor

            # Extract the pixels within the current block
            block_pixels = []
            for y in range(y_start, y_end):
                for x in range(x_start, x_end):
                    if grid[y, x]:
                        block_pixels.append((x, y))

            # If there are pixels in the block, calculate the centroid
            if block_pixels:
                centroid_x = sum(x for x, y in block_pixels) // len(block_pixels)
                centroid_y = sum(y for x, y in block_pixels) // len(block_pixels)
                centroid = (centroid_x, centroid_y)

                # Check if the centroid is near a wall
                if is_near_wall(centroid_x, centroid_y, wall_distance):
                    continue  # Skip this centroid if it is too close to a wall

                # Map the original pixels to this centroid
                if centroid not in centroids:
                    centroids[centroid] = set()
                centroids[centroid].update(block_pixels)

    return centroids


def add_hallways_graph(G, image_path, hallways_segments, row_factor=30, col_factor=30):
    """
    Create a NetworkX graph from centroids where each centroid is a node, and 
    there is an edge between two centroids if their blocks touch each other.
    
    Args:
        centroids (dict): Dictionary with keys as centroids (x, y) and values as
            sets of original pixels.
        row_factor (int): Factor by which the number of rows was reduced.
        col_factor (int): Factor by which the number of columns was reduced.
    
    Returns:
        G (networkx.Graph): Graph where nodes are centroids and edges exist 
            between touching blocks.
    """
    # Get image dimensions
    image = cv2.imread(image_path)
    image_height, image_width, _ = image.shape

    # Reduce hallways pixels density
    centroids = reduce_pixel_density(hallways_segments, image_width, 
                                     image_height, row_factor, col_factor)
    
    # Add nodes to global graph
    nodes_list = []
    for centroid in centroids.keys():
        hallway_node = node(id= G.next_id)
        hallway_node.set_coordinates(centroid[0], centroid[1])
        hallway_node.set_type("connective")
        hallway_node.set_label("hallway")
        nodes_list.append(hallway_node)
        G.add_node(hallway_node)
    
    # Add edges for adjacent centroids
    for i in range(len(nodes_list)):
        x1, y1 = nodes_list[i].get_coordinates() 
        for j in range(i + 1, len(nodes_list)):
            x2, y2 = nodes_list[j].get_coordinates()

            # Check if both centroids are adjacent
            if (abs(x1 - x2) <= col_factor) and (abs(y1 - y2) <= row_factor):
                G.add_edge(nodes_list[i], nodes_list[j])
    
    return G, centroids
