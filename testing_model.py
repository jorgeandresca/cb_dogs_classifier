from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # This will hide those Keras messages

"""
    Model input (299, 299, 3)
"""

from keras.models import load_model
from PIL import Image
import numpy as np
from helper import acc_classes


model = load_model('models/0.823.h5')
img_pil = Image.open("data/test/toy_poodle_1.jpg")
dataset_dir = 'data/dataset'

# Ensure the image is RGB since we need 3 channels
if img_pil.mode != "RGB":
    img_pil = img_pil.convert('RGB')

# Resize it
img_pil = img_pil.resize((299, 299))

# Convert to Array and ensure its dimension matches the Input of the model
img_arr = np.array(img_pil)  # Image -> Array
img_arr = np.expand_dims(img_arr, axis=0)  # (299,299,3) -> (1, 299,299,3)


# Prediction
prediction = (model.predict(img_arr, verbose=1))[0]

# Getting list of results
m = max(prediction)
acc_index = [sorted( [(x,i) for (i,x) in enumerate(prediction)], reverse=True )[:5]][0]
prediction = acc_classes('data/dataset', acc_index)

print(prediction)