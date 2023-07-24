import tensorflow as tf
from keras.optimizers import Adam
import json

from data_util import ImageDataset
from layer_architecture import DEFAULT_LAYERS
import pandas as pd

from datetime import datetime
from loguru import logger

import sys
import os

model_path = 'models/Model %s' % datetime.now()
os.mkdir(model_path)

train_log_file = open(model_path + '/train_log.txt', 'a+')

sys.stdout = train_log_file
sys.stderr = train_log_file

logger.add(model_path + '/train_log.txt')
logger.info('Read data: ./data/data.csv')
dataset = ImageDataset(pd.read_csv('data/data.csv').sample(frac=0.001, random_state=42))

data = dataset.split()

model = tf.keras.Sequential(DEFAULT_LAYERS)

opt = Adam(lr=0.000001)
model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

logger.info('Training started. # Epochs = 1')
results = model.fit(data['train_images'], data['train_labels'], epochs=1,
                    validation_data=(data['val_images'], data['val_labels']))
logger.info('Training finished. Evaluating model...')

test_loss, test_acc = model.evaluate(data['test_images'],  data['test_labels'], verbose=2)

print('ACCURACY', test_acc)
logger.info('Test loss: {}, test accuracy: {}', test_loss, test_acc)

model.save(model_path + '/model')
with open(model_path + '/epoch_results.json', 'w') as results_log_file:
    json.dump(results.history, results_log_file, indent=4)

sys.stdout.close()
sys.stderr.close()
train_log_file.close()
