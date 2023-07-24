import os
import pickle
import pandas as pd

from PIL import Image
import numpy as np
import yaml

from datetime import datetime


class Classifier:
    def __init__(self):
        df = pd.read_csv('../datasets/deep_fashion/classes.csv')
        self.classes = df['class'].tolist();

        # self.classes = params['classes']
        self.model_file = open('./models/Model %s/models' % self.__get_latest_model(), 'rb')

        self.model = pickle.load(self.model_file)

    def classify(self, img: Image):
        img = img.resize((256, 256))
        example = np.array(img)

        label = self.model.predict(example)
        return self.classes[np.argmax(label)]

    def __get_latest_model(self):
        model_paths = os.listdir('models/')
        model_paths = [path.split()[1] + ' ' + path.split()[2] for path in os.listdir('models/')]
        return str(max([datetime.strptime(path, '%Y-%m-%d %H:%M:%S.%f') for path in model_paths]))


classifier = Classifier()
classifier.classify(Image.open('../api/resources/downloads/mazapan.png'))