import webcolors
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

from scipy.spatial import KDTree
from webcolors import (
    CSS3_HEX_TO_NAMES,
    hex_to_rgb,
)

def img_recenter(img, height, width):
    return img[(height//4):(3*height//4), (width//4):(3*width//4), :]

def rgb_to_name(rgb_tuple):
    # a dictionary of all the hex and their respective names in css3
    css3_db = CSS3_HEX_TO_NAMES
    names = []
    rgb_values = []
    for color_hex, color_name in css3_db.items():
        names.append(color_name)
        rgb_values.append(hex_to_rgb(color_hex))

    tree = KDTree(rgb_values)
    distance, index = tree.query(rgb_tuple)
    return names[index]

def fetch_colors(img_path):
    img = cv2.imread(img_path)

    height, width, dimension = \
        img.shape

    img = img_recenter(img, height, width)

    height, width, dimension = \
        img.shape

    img_array = np.reshape(img, [height * width, dimension] )

    model = KMeans(n_clusters=3)
    model.fit(img_array)

    u, c = np.unique(model.labels_, return_counts=True)
    x = np.argsort(c)
    x = x[::-1]

    colors = []
    for center in model.cluster_centers_[x]:
        center = [int(x) for x in center]
        hex_color = '#%02x%02x%02x' % (center[2], center[1], center[0])
        rgb_color = center[2], center[1], center[0]
        print('COLOR', hex_color)
        print(rgb_to_name(rgb_color))
        colors.append(rgb_to_name(rgb_color))

    return colors
