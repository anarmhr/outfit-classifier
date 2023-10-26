import numpy as np
from PIL import Image

arr = np.random.rand((32, 32, 3), )

img = Image.fromarray(arr)
img.show()
