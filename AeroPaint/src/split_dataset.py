# %% Getting images
import os
from shutil import move, rmtree
import random
import cv2

base_dir = '../../Dataset/'
training_base_dir = os.path.join(base_dir, 'training')
testing_base_dir = os.path.join(base_dir, 'testing')
split_size = 0.7


def split_data(CLASS_NAME, SOURCE_PATH, TRAINING_PATH, TESTING_PATH, SPLIT_SIZE):
    if not os.path.isdir(TRAINING_PATH):
        os.mkdir(TRAINING_PATH)
        
    if not os.path.isdir(TESTING_PATH):
        os.mkdir(TESTING_PATH)
    
    shape_dir_training = os.path.join(TRAINING_PATH, class_name)
    shape_dir_testing = os.path.join(TESTING_PATH, class_name)
    
    if not os.path.isdir(shape_dir_training):
        os.mkdir(shape_dir_training)
        
    if not os.path.isdir(shape_dir_testing):
        os.mkdir(shape_dir_testing)
    
    files = []
    for filename in os.listdir(SOURCE_PATH):
        file = SOURCE_PATH + '/' +  filename
        if os.path.getsize(file) > 0:
            img = cv2.imread(file)
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            _, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
            cv2.imwrite(file, thresh.reshape(*thresh.shape, 1))
            files.append(filename)
        else:
            print(filename + " is zero length, so ignoring.")

    training_length = int(len(files) * SPLIT_SIZE)
    testing_length = int(len(files) - training_length)
    shuffled_set = random.sample(files, len(files))
    training_set = shuffled_set[0:training_length]
    testing_set = shuffled_set[-testing_length:]

    for filename in training_set:
        this_file = SOURCE_PATH +'/' +  filename
        destination = shape_dir_training +'/' +  filename
        move(this_file, destination)

    for filename in testing_set:
        this_file = SOURCE_PATH + '/' + filename
        destination = shape_dir_testing + '/' +  filename
        move(this_file, destination)
    
    rmtree(SOURCE_PATH)

classes = ['circle', 'square', 'triangle']
for class_name in classes:
    source_dir = os.path.join(base_dir, class_name)
    split_data(class_name, source_dir, training_base_dir, testing_base_dir, split_size)
