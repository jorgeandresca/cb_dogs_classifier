from flask import Flask
from flask import jsonify
from app_classifier import classifier


app = Flask(__name__)
app.register_blueprint(classifier, url_prefix="/classifier")


print("\n\n\n\n")
print("*** App is loaded")
print("\n\n\n\n")


@app.route("/", methods=['GET', 'POST'])
def home():
    return jsonify("Working")
