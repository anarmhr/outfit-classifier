import random

import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from loguru import logger
from sklearn.model_selection import train_test_split


class DataGenerator:
    def __init__(self, params: dict()):
        self.params = params

    def generate_data(self) -> dict():
        logger.info('Read data from {}', self.params['data-path'])

        skip = sorted(random.sample(range(1, 10000), 10000 - 20))

        print('SKIP', skip)
        image_data = pd.read_csv(self.params['data-path'], usecols=['path', 'label'], skiprows=skip)

        image_data['path'] = './data/img/' + image_data['path']

        train_data, test_data = train_test_split(image_data,
                                                 test_size=self.params['test-size'],
                                                 random_state=self.params['random-state'])
        print(train_data)
        data_generator = ImageDataGenerator(
            rescale=self.params['rescale'],
            rotation_range=self.params['rotation-range'],
            width_shift_range=self.params['width-shift-range'],
            height_shift_range=self.params['height-shift-range'],
            shear_range=self.params['shear-range'],
            zoom_range=self.params['zoom-range'],
            horizontal_flip=self.params['horizontal-flip'],
            vertical_flip=self.params['vertical-flip'],
            brightness_range=self.params['brightness-range']
        )

        img_res = tuple(self.params['input-shape'])
        batch_size = self.params['batch-size']

        train_generator = data_generator.flow_from_dataframe(
            dataframe=train_data,
            x_col='path',
            y_col='label',
            target_size=img_res,
            batch_size=batch_size,
            class_mode='raw',
            shuffle=True
        )

        test_generator = data_generator.flow_from_dataframe(
            dataframe=test_data,
            x_col='path',
            y_col='label',
            target_size=img_res,
            batch_size=batch_size,
            class_mode='raw',
            shuffle=True
        )

        return {
            'train-generator': train_generator,
            'test-generator': test_generator
        }
