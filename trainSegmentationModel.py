import numpy as np 
import os 
import keras 
from keras.utils import np_utils
from keras.optimizers import *
import h5py
import tensorflow as tf 

from models.segModel import createSegModel
from models.model import createModel
from utils.dataUtils import load_h5, rotate_point_cloud, jitter_point_cloud


# Parameters
num_points = 1024
k = 4
LEARNING_RATE = 1e-1
BATCH_SIZE = 32
EPOCHS = 50
WEIGHTS_PATH = './seg_weights'

# Creating Weights Path directory
if not os.path.exists(WEIGHTS_PATH):
    os.makedirs(WEIGHTS_PATH)

model, seg_part1 = createModel(num_points, k)
model = createSegModel(model, seg_part1, num_points, k)

# Loading training data
path = os.path.dirname(os.path.realpath(__file__))
train_path = os.path.join(path, "data", "SegPrepTrainData")
filenames = [d for d in os.listdir(train_path)]

print("Path For Training Data ", train_path)
print("Files ", filenames)

train_points = None 
train_labels = None 

for d in filenames :
    cur_points, cur_labels = load_h5(os.path.join(train_path, d))

    if train_labels is None or train_points is None:
        train_labels = cur_labels
        train_points = cur_points

    else:
        train_labels = np.hstack((train_labels, cur_labels))
        train_points = np.hstack((train_points, cur_points))

train_points_r = train_points.reshape(-1, num_points, 3)
train_labels_r = train_labels.reshape(-1, num_points, k)

# Loading testing data
test_path = os.path.join(path, "data", "SegPrepTestData")
filenames = [d for d in os.listdir(test_path)]

print("Path For Testing Data ", test_path)
print("Files ", filenames)

test_points = None 
test_labels = None 

for d in filenames :
    cur_points, cur_labels = load_h5(os.path.join(test_path, d))

    if test_labels is None or test_points is None:
        test_labels = cur_labels
        test_points = cur_points

    else:
        test_labels = np.hstack((test_labels, cur_labels))
        test_points = np.hstack((test_points, cur_points))

test_points_r = test_points.reshape(-1, num_points, 3)
test_labels_r = test_labels.reshape(-1, num_points, k)

train_points_rotate = rotate_point_cloud(train_points_r)
train_points_jitter = jitter_point_cloud(train_points_rotate)

# Compiling Model
optimizer = Adam(lr = LEARNING_RATE)
model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])

callbacks = [
            keras.callbacks.TensorBoard(log_dir='./logs',
                                        histogram_freq=0, write_graph=True, write_images=False),
             keras.callbacks.ModelCheckpoint(os.path.join(WEIGHTS_PATH, 'weights{epoch:08d}.h5'),
                                    verbose=0, save_weights_only=True)]
# Training Model 
model.fit(train_points_jitter, train_labels_r, batch_size = BATCH_SIZE, 
                epochs = EPOCHS, verbose = 1, validation_data = (test_points_r, test_labels_r), 
                callbacks = callbacks)
