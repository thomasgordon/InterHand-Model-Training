import math

def crop_landmarks(crop, landmarks, resized_size):
    left, top, right, bottom = crop
    resized_width, resized_height = resized_size
    final_landmarks = []

    if resized_width == 0 or resized_height == 0:
        return []
    else:
        x_ratio = 320/resized_width
        y_ratio = 320/resized_height

    for landmark in landmarks:
        x, y, v = landmark
        if x < left or x > right or y < top or y > bottom:
            final_landmarks.append([x, y, 0])
        else:
            x, y, v = landmark
            scaled_x = (x-left) * x_ratio
            scaled_y = (y-top) * y_ratio
            final_landmarks.append([scaled_x, scaled_y, v])


    return final_landmarks