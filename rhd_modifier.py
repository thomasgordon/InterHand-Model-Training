#This file creates augmentations of each image, randomly cropping images, and creates a new .pickle file for storing the landmark data. This is to avoid the hand detection model just predicting a square in the middle of the screen each time.

from loading import *
from image_utils import *

final_landmarks = []
path = './RHD_published_v2/training/color/'
initial_images, initial_landmarks = load_landmarks_and_images('training')
index = 1

for landmarks in initial_landmarks:
    final_landmarks.append(landmarks)

for i in range(0, len(initial_images) - 1):
    cropped_image, cropped_landmarks = crop_image(initial_images[i], initial_landmarks[i])

    visible_coordinates = [(x, y) for x, y, visibility in cropped_landmarks if visibility > 0]

    # Check if there are enough visible coordinates to create a bounding box
    if len(visible_coordinates) >= 4:
        file_name = str(i + 41257) + '.png'
        cropped_image.save(os.path.join(path, file_name))
        final_landmarks.append(cropped_landmarks)
        index+=1

with open('./RHD_published_v2/training/cropped_landmarks.pickle', 'wb') as file:
    pickle.dump(final_landmarks, file)