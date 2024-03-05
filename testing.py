from image_utils import *
from loading import *
from data_utils import *
import random
import os
import matplotlib.pyplot as plt

images, landmarks = load_landmarks_and_images('training')

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

def test_crop():
    for i in range(0, len(images)):
        cropped_image, cropped_landmarks = crop_image(images[i], landmarks[i])
        print(cropped_landmarks)

        try:
            min_x, max_x, min_y, max_y = get_bounding_box(cropped_landmarks)
        except:
            continue

        fig, ax = plt.subplots()
        ax.imshow(cropped_image)

        rect = plt.Rectangle((min_x - 10, min_y - 10), (max_x - min_x) + 20, (max_y - min_y) + 20, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        plt.show()




test_crop()