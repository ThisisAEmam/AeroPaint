# %% Getting images
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from model_arch import create_model

base_dir = '../../Dataset/'

training_dir = os.path.join(base_dir, 'training')
testing_dir = os.path.join(base_dir, 'testing')

LABELS = os.listdir(training_dir)
NUM_CLASSES = len(os.listdir(training_dir))

# %% CREATING THE DATASET
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1 / 255)

testing_datagen = ImageDataGenerator(rescale = 1 / 255)

train_generator = train_datagen.flow_from_directory(
        training_dir,
        target_size = (28, 28),
        class_mode = 'categorical',
        subset='training'
    )

test_generator = testing_datagen.flow_from_directory(
        testing_dir,
        target_size = (28, 28),
        class_mode = 'categorical'
    )

NUM_TRAINING = train_generator.samples
# %% CREATING MODEL
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, \
    InputLayer, Activation, Dropout
from tensorflow.keras.models import Sequential
import tensorflow_probability as tfp 

tfd = tfp.distributions
tfpl = tfp.layers

model = create_model(NUM_CLASSES)

# %% Compiling the model
from tensorflow.keras.optimizers import Adam

def neg_log_likelihood(y_true, y_pred):
    return -y_pred.log_prob(y_true)

model.compile(
        loss = neg_log_likelihood,
        optimizer = Adam(lr = 0.001),
        metrics = ['accuracy'],
        experimental_run_tf_function = False)

# %% Training our Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
history = model.fit(
        train_generator,
        validation_data = test_generator,
        epochs = 20,
        validation_steps = 50,
        callbacks = [
            ReduceLROnPlateau(patience=3, monitor='val_accuracy')])

model.save_weights('../../model/weights')
