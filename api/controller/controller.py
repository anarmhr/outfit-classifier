from flask import Flask, request
from classifier import Classifier
import requests
import color_detector
from flask_cors import CORS, cross_origin
from PIL import Image
from io import BytesIO
import base64
import time

app = Flask(__name__)

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

classifier = Classifier()

@app.route('/')
def index():
    return None


@app.route('/classify/by-path', methods=['POST'])
@cross_origin()
def classify_by_path():
    url = request.form['image_url']
    img_path = download_image(url)
    class_ = classifier.classify(img_path)
    print(img_path)
    colors = color_detector.fetch_colors(img_path)

    return {'class': class_,
            'colors': str(colors)}

@app.route('/classify/by-file', methods=['POST'])
@cross_origin()
def classify_by_file():
    img_file = request.files['image_data']
    img_file.save('images/' + img_file.filename)
    class_ = classifier.classify('images/' + img_file.filename)
    colors = color_detector.fetch_colors('images/' + img_file.filename)

    return {'class': class_,
            'colors': str(colors)}


@app.route('/classify', methods=['POST'])
@cross_origin()
def classify():
    img_data = request.form['image_data']

    if ',' in img_data:
        img_data = img_data.split(',')[1]

    print(img_data)
    # extension = mimetypes.guess_extension(img_data)

    img = Image.open(BytesIO(base64.b64decode(img_data)))
    timestamp = time.time_ns()
    filename = 'images/' + str(timestamp) + '.png'
    img.save(filename)

    class_ = classifier.classify(filename)
    print('CLASS', class_)
    colors = color_detector.fetch_colors(filename)

    return {'class': class_,
            'colors': str(colors)}


def download_image(url):
    out = requests.get(url, allow_redirects=True)
    name = url.split('/')[-1]
    img_path = './images/' + name

    open(img_path, 'wb').write(out.content)

    return img_path

