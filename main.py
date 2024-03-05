from data_utils import *
from image_utils import *
from loading import *

colour_images_train, training_landmarks = load_data('training')
colour_images_test, testing_landmarks = load_data('evaluation')