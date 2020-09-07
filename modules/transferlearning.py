from __future__ import print_function

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # This will hide those Keras messages

"""
    InvceptionV3 has input (299, 299, 3) ((in case the environment is configured to have the channel at the end)
"""


from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.models import load_model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from modules.helper import get_num_subfolders

# Training Configuration
image_width, image_height = 299, 299;
num_epochs = 300
batch_size = 32
training_size = 85  # 100 => 100%
dataset_dir = "../local/data/dataset"
pretrained_model = "../local/models/0.834.h5"
output_model = "../local/models/model.h5"
output_image = "../local/models/chart.png"
outputs = "../local/models"
num_classes = get_num_subfolders(dataset_dir)

# Data Augmentation. Creating variations of the images by rotating, shift up, down left, right, sheared, zoom in,
#   or flipped horizontally on vertical axis
#   This, replaces the original dataset by a new one.
# ImageDataGenerator accepts the original data, randomly transforms it, and returns only the new, transformed data.
# The network sees “new” images that it has never “seen” before at each and every epoch.
img_generator = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split= (100 - training_size)/100  # This will split the dataset in Training and Validation subsets.
    )

print(" Training set:")
train_generator = img_generator.flow_from_directory(
    dataset_dir,
    target_size=(image_width, image_height),
    batch_size=batch_size,
    #seed=42,
    subset="training"
)
print("  Validation set:")
validation_generator = img_generator.flow_from_directory(
    dataset_dir,
    target_size=(image_width, image_height),
    batch_size=batch_size,
    #seed=42,
    subset="validation"
)




# Preparing the model
base_model = None
model = None

if(pretrained_model == ""):
    # Load the pretrained model
    #   Exclude the final fully connected layer (include_top=false)
    base_model = InceptionV3(weights="imagenet", include_top=False)

    # Define a new classifier to attach to the pretrained model
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="relu")(x)
    output = Dense(num_classes, activation="softmax")(x)

    # Freeze all layers in the pretrained model (BASE_MODEL)
    for layer in base_model.layers:
        layer.trainable = False

    # Merge the pretrained model and the new FC classifier
    model = Model(inputs=base_model.input, outputs=output)

else:  # in case there is already a trained model.h5
    model = load_model(pretrained_model)

"""
#    print(model.summary()):
#
#        Total params: 24,131,585
#        Trainable params: 2,328,801 (2.098.176 from our Dense 1024, and 230.625 (1025 * 225) from our output layer)
#        Non-trainable params: 21,802,784
"""


# Compile
#   We use categorical_crossentropy since our model is trying to classify categorical result
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Callbacks
my_callbacks = [
    #tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5),
    tf.keras.callbacks.ModelCheckpoint(filepath=outputs + "/model.{epoch:02d}-{val_accuracy:.3f}.h5")
]

# Fit
hist = model.fit(
    train_generator,
    epochs=num_epochs,
    steps_per_epoch= len(train_generator.filepaths) // batch_size,
    validation_data=validation_generator,
    validation_steps= len(validation_generator.filepaths) // batch_size,
    callbacks=my_callbacks
)

# Evaluate the model
score_train = np.round(model.evaluate(train_generator, verbose=0), 3)
score_test = np.round(model.evaluate(validation_generator, verbose=0), 3)
print("Val loss: ", score_test[0])
print("Val accuracy: ", score_test[1])


# Saving model
model.save(output_model)


# Plot results
epoch_list = list(range(1, len(hist.history["accuracy"]) + 1))
plt.plot(epoch_list, hist.history["accuracy"], epoch_list, hist.history["val_accuracy"])
plt.legend(("Training Accuracy: " + str(score_train[1]), "Validation Accuracy: " + str(score_test[1])))
plt.savefig(output_image)
plt.show()

