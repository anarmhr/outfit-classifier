import base64
from io import BytesIO

import requests

from train.classifier import Classifier
from PIL import Image

classifier = Classifier()

DOWNLOAD_DEST = '../resources/downloads'


def classify_by_url(img_url):
    pass


def classify_by_image():
    pass


def classify_base64():
    pass


def base64_to_image(base64_data):
    if ',' in base64_data:
        img_data = base64_data.split(',')[1]

    return Image.open(BytesIO(base64.b64decode(img_data)))


def download_file(url):
    out = requests.get(url, allow_redirects=True)
    filename = url.split('/')[-1]
    file_path = DOWNLOAD_DEST + '/' + filename

    open(file_path, 'wb').write(out.content)
    return file_path
