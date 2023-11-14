import sys

import yaml
import random
import tensorflow as tf

import json
import argparse

#from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

import layer_architecture
import pandas as pd

from datetime import datetime
from loguru import logger

from data_generator import DataGenerator

import os

with open('./params/train-keras-params.yml', 'r') as params_file:
    params = yaml.safe_load(params_file)


params['model-dir'] = params['model-dir'] % datetime.now()
params['model-save-path'] = params['model-save-path'].replace('{{model-dir}}', params['model-dir'])
params['train-log-path'] = params['train-log-path'].replace('{{model-dir}}', params['model-dir'])
params['epoch-result-path'] = params['epoch-result-path'].replace('{{model-dir}}', params['model-dir'])
params['model-summary-path'] = params['model-summary-path'].replace('{{model-dir}}', params['model-dir'])

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--data', '-d', default=params['data-path'], help='Data path (must be a CSV file)')

# initialize paths and logger

os.mkdir(params['model-dir'])
logger.add(params['train-log-path'])

# generate data
data_generator = DataGenerator(params).generate_data()
train_generator = data_generator['train-generator']
test_generator = data_generator['test-generator']

# build model
model = tf.keras.Sequential(layer_architecture.layer_array[params['layer-architecture-index']])

#opt = Adam(lr=params['learning-rate'])
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# train model
logger.info('Training started. Epochs: {}', params['epochs'])
with tf.device('/device:GPU:0'):
    results = model.fit(train_generator, epochs=params['epochs'], steps_per_epoch=len(train_generator), verbose=1)

logger.info('Training finished. Evaluating model...')

test_loss, test_acc = model.evaluate(test_generator, verbose=2)
logger.info('Test loss: {}, test accuracy: {}', test_loss, test_acc)

model.save(params['model-save-path'], save_format='tf')

with open(params['epoch-result-path'], 'w') as results_log_file:
    json.dump(results.history, results_log_file, indent=4)

# Prepare Summary
logger.info('Preparing model summary...')

model_summary_text = []
model.summary(print_fn=lambda x: model_summary_text.append(x))
model_summary_text = "\n".join(model_summary_text)

with open(params['model-summary-path'], 'w') as model_summary_file:
    model_summary_file.write(model_summary_text)

logger.info('Model summary saved: {}', params['model-summary-path'])