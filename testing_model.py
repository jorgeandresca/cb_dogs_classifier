from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # This will hide those Keras messages

"""
    Model input (299, 299, 3)
"""

from keras.models import load_model
from PIL import Image
import numpy as np

model = load_model('models/model 2020.09.02.h5')
img_pil = Image.open("data/test/pelican.jpg")

# Ensure the image is RGB since we need 3 channels
if img_pil.mode != "RGB":
    img_pil = img_pil.convert('RGB')

# Resize it
img_pil = img_pil.resize((299, 299))

# Convert to Array and ensure its dimension matches the Input of the model
img_arr = np.array(img_pil)  # Image -> Array
img_arr = np.expand_dims(img_arr, axis=0)  # (299,299,3) -> (1, 299,299,3)

print(img_arr.shape)

# Prediction
prediction = (model.predict(img_arr, verbose=1))

print(prediction[0])
