from random import randint
from data_utils import *
from PIL import Image
import math

#! PARAMETERS
shape = (320, 320, 3)
image_xy = 320

def crop_image(image_path, landmarks):
    image = Image.open(image_path)
    image_width, image_height = image.size
    image_half = math.floor(min(image_width, image_height) / 2)

    crop1 = [randint(0, image_half), randint(0, image_half)]
    crop2 = [randint(crop1[0], 320), randint(crop1[1], 320)]

    left, top = crop1
    right, bottom = crop2

    cropped_image = image.crop((left, top, right, bottom))

    final_landmarks = crop_landmarks((left, top, right, bottom), landmarks, cropped_image.size)

    final_image = cropped_image.resize((320, 320))

    return final_image, final_landmarks

def rotate_image(image, landmarks):
    pass

def invert_image(image, landmarks):
    pass