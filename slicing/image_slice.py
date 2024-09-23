import matplotlib.pyplot as plt
import json

import pandas as pd
import numpy as np
from matplotlib import patches

import albumentations as A
from functools import partial
import os
from PIL import Image
from multiprocessing import Pool

from .dataset import DatasetAdaptor, ImageSliceDetectionDataset

root = os.path.dirname(__file__) + "/../"

"""
Given the height and width of an image, calculates how to divide the image into
overlapping slices according to the height and width provided. These slices are returned
as bounding boxes in xyxy format.

parameters
- image_height: Height of the original image, int
    (required)
- image_width: Width of the original image, int
    (required)
- slice_height: Height of each slice, int
    (default 512)
- slice_width: Width of each slice, int
    (default 512)
- overlap_height_ratio: Fractional overlap in height of each slice 
                        (e.g. an overlap of 0.2 for a slice of size 
                        100 yields an overlap of 20 pixels), float
    (default 0.2)
- overlap_width_ratio: Fractional overlap in width of each slice 
                       (e.g. an overlap of 0.2 for a slice of size 
                       100 yields an overlap of 20 pixels,), float
    (default 0.2)

returns 
    list of bounding boxes in xyxy format, (int * int * int * int) list
"""
def calculate_slice_bboxes(
    image_height: int,
    image_width: int,
    slice_height: int = 512,
    slice_width: int = 512,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
):
    slice_bboxes = []
    y_max = y_min = 0

    y_overlap = int(overlap_height_ratio * slice_height)
    x_overlap = int(overlap_width_ratio * slice_width)

    while y_max < image_height:
        x_min = x_max = 0
        y_max = y_min + slice_height

        while x_max < image_width:
            x_max = x_min + slice_width

            if y_max > image_height or x_max > image_width:
                xmax = min(image_width, x_max)
                ymax = min(image_height, y_max)
                xmin = max(0, xmax - slice_width)
                ymin = max(0, ymax - slice_height)

                slice_bboxes.append([xmin, ymin, xmax, ymax])
            else:
                slice_bboxes.append([x_min, y_min, x_max, y_max])
            
            x_min = x_max - x_overlap

        y_min = y_max - y_overlap

    return slice_bboxes

"""
Returns a dataframe that specifies the width and height
of every image that needs to be sliced over

parameters
- images_path: location of images, str
    (required)
- file_names: names of images to slice, str list
    (required)

returns 
    dataframe with width and heights, pd.DataFrame
"""
def get_image_sizes_df(images_path, file_names):
    pool = Pool()
    image_sizes = pool.map(partial(get_image_size, images_path=images_path), file_names)
    sizes_df = pd.DataFrame(image_sizes)
    return sizes_df

"""
returns a dictionary with the height and width of the image 
at the location given

parameters
- file_name: name of image file, str
- images_path: location of image file, str

returns
    dictionary with 'file_name' 'image_height' 'image_width', dict
"""
def get_image_size(file_name, images_path):
    image = Image.open(f"{images_path}/{file_name}")
    w, h = image.size
    return {"file_name": file_name, "image_height": h, "image_width": w}

"""
given a series of images in a dataframe, creates slices of the given size
to conver all of each image

parameters
- images_path: location of image files, str
    (required)
- remote_df: dataframe with image file names to slice, pd.DataFrame
    (required)
- slice_height: height of each slice, int
    (default 250)
- slice_width: width of each slice, int
    (default 250)
- overlap_height_ratio: area of overlap between each slice, float
    (default 0.2)
- overlap_width_ratio: area of overlap between each slice, float
    (default 0.2)

returns 
    dataframe of each slice's min and max x,y coordinates, pd.DataFrame
"""
def create_image_slices_df(
    images_path,
    remote_df,
    slice_height: int = 250,
    slice_width: int = 250,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2
):
    sizes_df = get_image_sizes_df(images_path, remote_df.file_name.unique())
    sizes_df["slices"] = sizes_df.apply(
        lambda row: calculate_slice_bboxes(
            row.image_height,
            row.image_width,
            slice_height,
            slice_width,
            overlap_height_ratio,
            overlap_width_ratio,
        ),
        axis=1,
    )

    slices_row_df = (
        sizes_df[["file_name", "slices"]]
        .explode("slices")
        .rename(columns={"slices": "slice"})
    )
    slices_row_df = pd.DataFrame(
        slices_row_df.slice.tolist(),
        columns=["xmin", "ymin", "xmax", "ymax"],
        index=slices_row_df.file_name,
    ).reset_index()

    image_slices_df = pd.merge(
        slices_row_df,
        sizes_df[["file_name", "image_height", "image_width"]],
        how="inner",
        on="file_name",
    )
    image_slices_df.reset_index(inplace=True)
    image_slices_df.rename(columns={"index": "slice_id"}, inplace=True)

    return image_slices_df

"""
given one image name, returns a dataframe of the slice name and the min, max x,y coordinates 
for all slices required to cover the image

parameters
- file: name of image file, str
    (required)
- images_path: folder location of image, str
    (default '../data')
- overlap_ratio: area of overlap in both x & y direction for each slice, float
    (default 0.2)
- size: width and height of each slice, int
    (default 250)
- name: directory for slice image file names, str
    (default '')

returns 
    dataframe with columns 'name' 'xmin' 'ymin' 'xmax' ymax', pd.DataFrame
"""
def slice_images(file, images_path=root + "data/", overlap_ratio=0.2, size=250, name=""):
    os.makedirs(name, exist_ok=True)
    
    file_df = pd.DataFrame(columns=["file_name"])
    file_df["file_name"] = [file]

    df = create_image_slices_df(images_path, file_df, 
            overlap_width_ratio=overlap_ratio, overlap_height_ratio=overlap_ratio,
            slice_width=size, slice_height=size)
    ds = DatasetAdaptor(images_path, file_df)
    dataset = ImageSliceDetectionDataset(ds, df)

    panels = pd.DataFrame(columns=["name", "xmin", "ymin", "xmax", "ymax"])

    for idx in df["slice_id"]:
        image, bounds = dataset[idx]
        im = Image.fromarray(image)
        im.save(f"{name}panel-{idx}.png")

        panels.loc[len(panels.index)] = ([f"{name}panel-{idx}.png"] + bounds)
    
    return panels