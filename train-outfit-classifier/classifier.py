import os

import pandas as pd
import keras
from PIL import Image
import numpy as np

import tensorflow as tf


from datetime import datetime
from loguru import logger


class Classifier:
    def __init__(self):
        df = pd.read_csv('./data/classes.csv')
        self.classes = df['class'].tolist();

        self.model = keras.models.load_model('./models/Model 2023-07-27 15:02:23.093883/model')
        self.input_shape = self.model.layers[0].input_shape[1:3]

        print(self.input_shape)

    def classify(self, image: Image):
        img_tensor = tf.image.resize(np.array(image), self.input_shape)
        try:
            prediction = self.model.predict(np.expand_dims(img_tensor / 255, 0))
            return
        except Exception as e:
            logger.error('Error occurred while classifying image: {}', str(e))
            return

        return self.classes[np.argmax(prediction)]

    def __get_latest_model(self):
        model_paths = os.listdir('models/')
        model_paths = [path.split()[1] + ' ' + path.split()[2] for path in os.listdir('models/')]
        return str(max([datetime.strptime(path, '%Y-%m-%d %H:%M:%S.%f') for path in model_paths]))



