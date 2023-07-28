from datetime import datetime

from PIL import Image
import pandas as pd
import keras
import numpy as np
import tensorflow as tf
from logger import logger
import os

import sys
sys.path.append('../../train-outfit-classifier')
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

from image_util import download_image, base64_to_image


class Classifier:
    def __init__(self):
        df = None
        try:
            df = pd.read_csv('../train-outfit-classifier/data/classes.csv')
        except Exception as e:
            logger.error('Error occurred while reading classes file: {}', str(e))
            quit(1)

        self.classes = df['class'].tolist();

        try:
            self.model = keras.models.load_model(
                '../train-outfit-classifier/models/Model 2023-07-27 15:02:23.093883/model')
        except Exception as e:
            logger.error('Error occurred while importing model: {}', str(e))
            return

        self.input_shape = self.model.layers[0].input_shape[1:3]

        print(self.input_shape)

    def classify(self, image: Image):
        img_tensor = tf.image.resize(np.array(image), self.input_shape)

        try:
            prediction = self.model.predict(np.expand_dims(img_tensor / 255, 0))
        except Exception as e:
            logger.error('Error occurred while classifying image: {}', str(e))
            return

        return self.classes[np.argmax(prediction)]

    def __get_latest_model(self):
        model_paths = [path.split()[1] + ' ' + path.split()[2] for path in os.listdir('models/')]
        return str(max([datetime.strptime(path, '%Y-%m-%d %H:%M:%S.%f') for path in model_paths]))


classifier = Classifier()


def classify_by_url(img_url):
    return classifier.classify(download_image(img_url))


def classify_by_image(image: Image):
    return classifier.classify(image)


def classify_base64(base64_data):
    return classifier.classify(base64_to_image(base64_data))


print(classifier.model)