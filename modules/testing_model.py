from __future__ import print_function

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # This will hide those Keras messages

"""
    Model input (1, 299, 299, 3)
"""
from keras.models import load_model
from PIL import Image
import numpy as np
from keras.applications.inception_v3 import preprocess_input
from modules.helper import acc_classes


# ---------- Configuration ---------
img_pil = Image.open("../local/data/test/toy_poodle_2.jpg")
dataset_dir = "../local/data/dataset"
model = load_model("../local/models/0.828.h5")


# Ensure the image is RGB since we need 3 channels
if img_pil.mode != "RGB":
    img_pil = img_pil.convert("RGB")

# Convert to Array and ensure its dimension matches the Input of the model
img_pil = img_pil.resize((299, 299))
img_arr = np.array(img_pil)  # Image -> Array
img_arr = np.expand_dims(img_arr, axis=0)  # (299,299,3) -> (1, 299,299,3)

# Normalizing the pixels
img_arr = preprocess_input(img_arr)


# Prediction
prediction = (model.predict(img_arr, verbose=1))[0]
acc_index = [sorted( [(x,i) for (i,x) in enumerate(prediction)], reverse=True )[:5]][0]  # top 5 predictions
prediction = acc_classes("classes.json", acc_index)

print(prediction)
