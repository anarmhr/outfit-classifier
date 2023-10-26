import json
import os
import sys

sys.path.append(os.getcwd() + '/modules')
from flask import Flask, request, make_response
from flask_cors import CORS, cross_origin
from modules.responses import ServiceResponse, Status, ClassificationResponse

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, ''))
sys.path.append(parent_dir)

from modules.classification_service import classify_by_url, classify_base64
from modules.logger import logger
from modules.responses import CustomJSONEncoder

app = Flask(__name__)

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


def json_respond(serializable_obj):
    return json.dumps(serializable_obj, cls=CustomJSONEncoder)


@app.route('/')
def index():
    return make_response({'field': 'value'})


@app.route('/classify/by-url', methods=['POST'])
@cross_origin()
def get_by_url():
    image_url = request.form['image_url']
    try:
        outfit_category = classify_by_url(image_url)
        result = ClassificationResponse(outfit_category, [])
    except Exception as e:
        logger.error('Error occurred while classifying image by url: {}, {}', image_url, str(e))
        error_message = 'Error occurred while classifying image by url'
        return ServiceResponse(status=Status.FAIL, message=error_message).make_response()

    return ServiceResponse(status=Status.OK, data=result).make_response()


@app.route('/classify', methods=['POST'])
@cross_origin()
def classify_by_base64():
    image_data = request.form['image_data']

    try:
        outfit_category = classify_base64(image_data)
        result = ClassificationResponse(outfit_category, ['red', 'blue'])
    except Exception as e:
        logger.error('Error occurred while classifying image by base64: {}', str(e))
        error_message = 'Error occurred while classifying image by url'
        json_respond(ServiceResponse(status=Status.OK, message=error_message))

    return json_respond(ServiceResponse(status=Status.OK, data=result))

if __name__ == '__main__':
    app.run(debug=True)