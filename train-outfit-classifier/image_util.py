from tensorflow import keras

import numpy as np
import keras_cv
import matplotlib.pyplot as plt
from PIL import Image as im
from skimage.color import rgb2gray
import skimage

from matplotlib import image as mpimg


def load_image(img_path, target_size):
    return img_to_array(keras.utils.load_img(img_path, target_size=target_size))


def to_grayscale(img_array):
    return skimage.img_as_ubyte(rgb2gray(img_array)) / 255.0
    # return rgb2gray(img_array)

def img_to_array(img):
    return np.array(img)


def array_to_img(array):
    return im.fromarray(array)


def display_image(img_array):
    image = array_to_img(img_array)
    plt.imshow(image, cmap=plt.cm.binary)
    plt.show()

