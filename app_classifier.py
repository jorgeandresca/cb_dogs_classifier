from flask import jsonify
from flask import request
from flask import Blueprint, render_template
from PIL import Image
from keras.applications.inception_v3 import preprocess_input
from keras.applications.imagenet_utils import decode_predictions
from keras.models import load_model
import numpy as np
from io import BytesIO
import requests


classifier = Blueprint("classifier", __name__)


@classifier.route("/", methods=['GET', 'POST'])
def home():
    return jsonify("Working")



@classifier.route("/classify_image", methods=['POST'])
def classify_image():
    """
        Receives an image. Form-data
    """
    print("\n\n\nclassify_image")

    # Retrieve and open the file
    file = request.files['image']
    image = Image.open(file.stream)

    return classify(image)


@classifier.route("/classify_url", methods=['POST'])
def classify_url():
    """
        Receives a URL. Json
    """
    print("\n\n\nclassify_url")
    image_url = request.get_json()["image_url"]
    print("\nimage_url: " + image_url + "\n\n\n")
    response = requests.get(image_url)  # Download the image from URL
    image = Image.open(BytesIO(response.content))

    return classify(image)


model = None  # This model variable is here to hold the model out of the API endpoint


def classify(image):
    global model  # This global model must be placed here, under the API

    if model is None:
        print("\n\n *** Loading the model \n\n")
        model = load_model("model.h5")  # Input (None, 299,299,3)
    else:
        print("\n\n *** Model is already loaded \n\n")

    # Ensure the image is RGB since we need 3 channels
    if image.mode != "RGB":
        image = image.convert('RGB')

    # Resize it
    image = image.resize((299, 299))

    # Convert to Array and ensure its dimension matches the Input of the model
    image = np.array(image)  # Image -> Array
    image = np.expand_dims(image, axis=0)  # (299,299,3) -> (1, 299,299,3)
    image = preprocess_input(image)

    # Prediction
    prediction = decode_predictions(model.predict(image, verbose=1))
    prediction = prediction[0]

    all_predictions = []
    for i in range(len(prediction)):
        class_predicted = {"class": prediction[i][1], "accuracy": float(prediction[i][2])}
        all_predictions.append(class_predicted)


    return jsonify(all_predictions)
