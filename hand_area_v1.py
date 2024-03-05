import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Flatten, Dense
from keras import models, layers
from keras.models import Model
from keras.optimizers import Adam
import os
import pickle
import random
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


# !HYPERPARAMETERS
image_xy = 320
batch_size = 16
input_shape = (320, 320, 3)
epochs = 5

def create_model():
    base_model = tf.keras.applications.MobileNetV2(input_shape=(320, 320, 3),
                                                   include_top=False,
                                                   weights='imagenet')
    base_model.trainable = False

    # Add the fully connected layers
    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4, activation='relu')  # Output layer for bounding box coordinates
    ])

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse',  # Mean Squared Error for regression
                  metrics=[tf.keras.metrics.MeanAbsoluteError()])  # Example metric: IoU

    return model

def load_landmarks_and_images(set):
    # Load landmark data
    with open(os.path.join('./RHD_published_v2/', set, 'anno_%s.pickle' % set), 'rb') as f:
        landmark_data_list = pickle.load(f)

    # Load and sort image file paths
    image_directory = os.path.join('./RHD_published_v2/', set, 'color')
    image_files = sorted([os.path.join(image_directory, file)
                          for file in os.listdir(image_directory)
                          if file.endswith(".png")])

    # Initialize the lists for valid landmarks and image paths
    valid_landmarks = []
    valid_image_files = []
    temp = 0
    # Iterate through each set of landmark data
    for i in range(0, len(landmark_data_list)):
        # Extract the uv_vis data for the current landmark data
        uv_vis_coords = landmark_data_list[i].get('uv_vis')

        # Initialize lists for storing landmarks and visibility for the current data point
        landmarks_uv = []

        if any(x < 0 or x > 320 for point_coords in uv_vis_coords for x in point_coords[:2]):
            temp+=1
            continue

        # Count the number of visible landmarks
        num_visible_landmarks = sum(1 for coords in uv_vis_coords if coords[2] > 0)

        # Check if there are at least 2 visible landmarks
        if num_visible_landmarks >= 2:
            for point_coords in uv_vis_coords:
                uv = point_coords[:2]
                vis = point_coords[2]
                for coord in uv:
                    landmarks_uv.append(coord / 320)
                landmarks_uv.append(vis)  # Append visibility along with x, y coordinates

            # Convert the flat list to a list of tuples (x, y, vis)
            landmarks_tuples = [(landmarks_uv[j], landmarks_uv[j + 1], landmarks_uv[j + 2]) for j in range(0, len(landmarks_uv), 3)]

            # Append the list of tuples to the valid_landmarks list
            valid_landmarks.append(landmarks_tuples)
            valid_image_files.append(image_files[i])
    
    print('skipped:', temp)
    print('using:', len(valid_image_files))

    # Make sure we have the same number of valid images and landmarks
    assert len(valid_image_files) == len(valid_landmarks), "Images and landmarks count does not match."
    return valid_image_files, valid_landmarks

def get_bounding_box(landmarks):

    # Filter out coordinates that are visible
    visible_coordinates = [(x, y) for x, y, visibility in landmarks if visibility > 0]

    # Check if there are enough visible coordinates to create a bounding box
    if len(visible_coordinates) >= 2:
        # Find minimum and maximum x and y coordinates
        min_x = min(visible_coordinates, key=lambda p: p[0])[0]
        max_x = max(visible_coordinates, key=lambda p: p[0])[0]
        min_y = min(visible_coordinates, key=lambda p: p[1])[1]
        max_y = max(visible_coordinates, key=lambda p: p[1])[1]

        # Display or use the bounding box coordinates
        return min_x, max_x, min_y, max_y
    else:
        print("Not enough visible coordinates to form a bounding box.")

def data_generator(image_files, landmarks, batch_size):
    num_samples = len(image_files)
    while True:
        for offset in range(0, num_samples, batch_size):
            batch_images = []
            batch_bounding_boxes = []
            for i in range(offset, min(offset + batch_size, num_samples)):
                image_path = image_files[i]
                with Image.open(image_path) as img:
                    img = np.array(img)
                    img = img.astype(np.float32) / 255.0
                    batch_images.append(img)

                # Get bounding box
                bounding_box = get_bounding_box(landmarks[i])
                batch_bounding_boxes.append(bounding_box)

            yield np.array(batch_images), np.array(batch_bounding_boxes)





colour_images_train, training_landmarks = load_landmarks_and_images('training')
colour_images_test, testing_landmarks = load_landmarks_and_images('evaluation')

model = create_model()

# Define the training and validation generators
train_generator = data_generator(colour_images_train, training_landmarks, batch_size)
val_generator = data_generator(colour_images_test, testing_landmarks, batch_size)



history = model.fit(train_generator,
                    steps_per_epoch=len(colour_images_train) // batch_size,
                    epochs=epochs,
                    validation_data=val_generator,
                    validation_steps=len(colour_images_test) // batch_size)


model.save('hand_detector.h5')