import os
import numpy
import pickle

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
                    landmarks_uv.append(coord)
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

def expand_array(initial_images, initial_landmarks):
    pass

def load_data(set) :
    initial_images, initial_landmarks = load_landmarks_and_images(set)