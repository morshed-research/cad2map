import os
import numpy as np
import cv2
import pandas as pd
from PIL import Image

"""
Takes inputs as 8 points and Returns cropped, masked image with a white background

parameters
- pts
    (required)
- image: image to infer on, cv2 image
    (required)

returns 
    ??
"""
def crop(pts, image):
    rect = cv2.boundingRect(pts)
    x,y,w,h = rect

    cropped = image[y:y+h, x:x+w].copy()

    pts = pts - pts.min(axis=0)
    mask = np.zeros(cropped.shape[:2], np.uint8)

    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
    dst = cv2.bitwise_and(cropped, cropped, mask=mask)
    bg = np.ones_like(cropped, np.uint8)*255
    cv2.bitwise_not(bg,bg, mask=mask)

    dst2 = bg + dst

    return dst2

"""
generates PIL image of bounding box portion

parameters
- bbox: bounding boxes to crop to
- image: image to crop, cv2 image
"""
def generate_image(bbox, image):
    if np.all(bbox) > 0:
        try:
            word = crop(bbox, image)
            color_coverted = cv2.cvtColor(word, cv2.COLOR_BGR2GRAY)

            return Image.fromarray(color_coverted).convert('L') 
        except:
            return None