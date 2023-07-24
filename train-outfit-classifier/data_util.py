import random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as im
from sklearn.model_selection import train_test_split
from tensorflow import keras


class ImageDataset:
    def __init__ (self, dataframe):
        self.dataframe = dataframe
        self.size = self.dataframe.shape[0]

    def get_labels(self):
        return np.array(self.dataframe['label'])

    def display_image(self, index=None):
        if not index:
            index = random.randint(0, self.dataframe.shape[0])

        im.open('./data/img/%s' % self.dataframe['path'][index]).show()

    def display_images(self, n=6):
        shape = (n // 3 if n % 3 == 0 else n // 3 + 1), 3
        figure = plt.figure(figsize=shape)

        for i in range(1, np.prod(shape) + 1):
            img_index = random.randint(0, self.dataframe.shape[0]);
            img = im.open('./data/img/%s' % self.dataframe['path'][img_index])

            figure.add_subplot(shape[0], shape[1], i)
            plt.imshow(img)
            plt.title(self.dataframe['path'][img_index].split('/')[0].split('_')[-1])

        figure.tight_layout(pad=0.5)
        plt.show()

    def split(self, train_ratio=.2, val_ratio=.2, test_ratio=.2):
        self.dataframe = self.dataframe.head(n=1000)

        train_data, test_data = \
            train_test_split(self.dataframe, test_size=0.2)

        train_data, val_data = \
            train_test_split(train_data, test_size=0.2)

        train_images = np.array([np.asarray(keras.utils.load_img ('data/img/%s' % path, target_size=(256, 256))) for path in train_data['path']]) / 255
        train_labels = np.array(train_data['label'])

        val_images = np.array([np.asarray(keras.utils.load_img('data/img/%s' % path, target_size=(256, 256))) for path in val_data['path']]) / 255
        val_labels = np.array(val_data['label'])

        test_images = np.array([np.asarray(keras.utils.load_img('data/img/%s' % path, target_size=(256, 256))) for path in test_data['path']]) / 255
        test_labels = np.array(test_data['label'])

        return {'train_images': train_images, 'train_labels': train_labels,
                'val_images': val_images, 'val_labels': val_labels,
                'test_images': test_images, 'test_labels': test_labels}


