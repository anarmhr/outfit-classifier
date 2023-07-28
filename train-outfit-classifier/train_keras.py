import pickle

import numpy as np
import tensorflow as tf

import json

from data_util import ImageDataset
import layer_architecture
import pandas as pd

from datetime import datetime
from loguru import logger

import os

# initialize paths and logger
MODEL_DIR = 'models/Model %s' % datetime.now()
MODEL_PATH = MODEL_DIR + '/model'
TRAIN_LOG_PATH = MODEL_DIR + '/train.log'
EPOCH_RESULTS_PATH = MODEL_DIR + '/epoch_results.json'
DATA_PATH = 'data/data.csv'
os.mkdir(MODEL_DIR)

logger.add(TRAIN_LOG_PATH)
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=MODEL_DIR + '/train.log')

# load data
logger.info('Read data from {}', DATA_PATH)
dataset = ImageDataset(pd.read_csv(DATA_PATH).sample(frac=0.001, random_state=42))

data = dataset.split()

# build model
model = tf.keras.Sequential(layer_architecture.DEFAULT_LAYERS)

# opt = Adam(lr=0.000001)
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# train model
logger.info('Training started. # Epochs = 1')
results = model.fit(data['train_images'], data['train_labels'], epochs=1,
                    validation_data=(data['val_images'], data['val_labels']))

image = np.array(tf.keras.utils.load_img('../api/resources/daria.png', target_size=(64, 64)))
prediction = model.predict(np.expand_dims(image / 255, 0))
logger.info(prediction)
logger.info("{}     {}", max(prediction[0]), np.argmax(prediction[0]))

logger.info('Training finished. Evaluating model...')

test_loss, test_acc = model.evaluate(data['test_images'], data['test_labels'], verbose=2)
logger.info(model.summary())
print('ACCURACY', test_acc)
logger.info('Test loss: {}, test accuracy: {}', test_loss, test_acc)

model.save(MODEL_PATH, save_format='tf')

with open(EPOCH_RESULTS_PATH, 'w') as results_log_file:
    json.dump(results.history, results_log_file, indent=4)
