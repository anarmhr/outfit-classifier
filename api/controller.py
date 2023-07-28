import json
import os
import sys

sys.path.append(os.getcwd() + '/modules')
from flask import Flask, request
from flask_cors import CORS, cross_origin
from modules.service_response import ServiceResponse, Status, ClassificationResponse

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, ''))
sys.path.append(parent_dir)

from modules.classification_service import classify_by_url, classify_base64
from modules.logger import logger

app = Flask(__name__)

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

LOG_PATH = '../logs/app.log'
logger.add(LOG_PATH, rotation='10 MB')

@app.route('/')
def index():
    return None


@app.route('/classify/by-url', methods=['POST'])
@cross_origin()
def get_by_url():
    image_url = request.form['image_url']

    message = None
    data = None
    status = None

    try:
        outfit_category = classify_by_url(image_url)
        status = Status.OK.name
        data = ClassificationResponse(outfit_category, []).serialize()
    except Exception as e:
        logger.error('Error occurred while classifying image by url: {}, {}', image_url, str(e))
        message = 'Error occurred while classifying image by url'

    return json.dumps(ServiceResponse(status, message, data).serialize())


@app.route('/classify', methods=['POST'])
@cross_origin()
def classify_by_base64():
    image_data = request.form['image_data']

    message = None
    data = None
    status = None

    try:
        outfit_category = classify_base64(image_data)
        status = Status.OK
        data = ClassificationResponse(outfit_category, ['red', 'blue']).serialize()
    except Exception as e:
        logger.error('Error occurred while classifying image by base64: {}', str(e))
        message = 'Error occurred while classifying image by url'

    return json.dumps(ServiceResponse(Status.OK.name, message, data).serialize())
