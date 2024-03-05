import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Flatten, Dense
from keras.models import Model
from keras.optimizers import Adam
import os
import pickle
import random
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Load the Keras model
model_path = "/Users/tom/Desktop/uni/year_3/fyp/tatu/tf_model/hand_detector.h5"
model = tf.keras.models.load_model(model_path)

set = 'training'

with open(os.path.join('/Users/tom/Desktop/uni/year_3/fyp/tatu/tf_model/RHD_published_v2/', set, 'anno_%s.pickle' % set), 'rb') as fi:
    anno_all = pickle.load(fi)

for sample_id, anno in anno_all.items():
    # load data
    path = (os.path.join('/Users/tom/Desktop/uni/year_3/fyp/tatu/tf_model/RHD_published_v2/', set, 'color', '%.5d.png' % int(sample_id)))

    img = Image.open(path)
    img = img.resize((224, 224))
    mage = np.array(image)  # Convert to numpy array
    image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)

    random_index = random.randint(0, len(anno_all.items()) - 1)

    fig, ax = plt.subplots()
    ax.imshow(img)

    min_x, max_x, min_y, max_y = model.predict(img)

    rect = plt.Rectangle((min_x - 10, min_y - 10), (max_x - min_x) + 20, (max_y - min_y) + 20, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    # Show the plot
    plt.show()

