import base64
from io import BytesIO

import requests
from PIL import Image


def download_image(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for 4xx or 5xx status codes
        image = Image.open(BytesIO(response.content))
        return image
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {e}")
        return None


def base64_to_image(base64_data):
    if ',' in base64_data:
        img_data = base64_data.split(',')[1]

    return Image.open(BytesIO(base64.b64decode(img_data)))

