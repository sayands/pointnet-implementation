# Importing neccessary packages and libraries
import numpy as np 
import os 
import sys
sys.path.append('../')
from keras import optimizers
from keras.layers import * 
from keras.models import * 
from keras.utils.vis_utils import plot_model




def createClassModel(NUM_POINTS, k):
    # PointNet Implementation
    input_points = Input(shape = (NUM_POINTS, 3))

    # Input Transformation Net
    x = Conv1D(64, 1, activation = 'relu', input_shape = (NUM_POINTS, 3))(input_points)
    x = BatchNormalization()(x)
    x = Conv1D(128, 1, activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(1024, 1, activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool1D(pool_size = NUM_POINTS)(x)
    x = Dense(512, activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = Dense(9, weights = [np.zeros([256, 9]),  np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)])(x)
    input_T = Reshape((3, 3))(x)

    # Forward Net
    g = Lambda(mat_mul, arguments = { 'B' : input_T})(input_points)
    g = Conv1D(64, 1, input_shape = (NUM_POINTS, 3), activation = 'relu')(g)
    g = BatchNormalization()(g)
    g = Conv1D(64, 1, input_shape =(NUM_POINTS, 3), activation = 'relu')(g)
    g = BatchNormalization()(g)

    # Feature Transformation Net
    f = Conv1D(64, 1, activation = 'relu')(g)
    f = BatchNormalization()(f)
    f = Conv1D(128, 1, activation = 'relu')(f)
    f = BatchNormalization()(f)
    f = Conv1D(1024, 1, activation = 'relu')(f)
    f = BatchNormalization()(f)
    f = MaxPool1D(pool_size = NUM_POINTS)(f)
    f = Dense(512, activation = 'relu')(f)
    f = BatchNormalization()(f)
    f = Dense(256, activation = 'relu')(f)
    f = BatchNormalization()(f)
    f = Dense(64 * 64, weights = [np.zeros([256, 64 * 64]), np.eye(64).flatten().astype(np.float32)])(f)
    feature_T = Reshape((64, 64))(f)

    # Forward Net
    g = Lambda(mat_mul, arguments = { 'B' : feature_T})(g)
    g = Conv1D(64, 1, activation = 'relu')(g)
    g = BatchNormalization()(g)
    g = Conv1D(128, 1, activation = 'relu')(g)
    g = BatchNormalization()(g)
    g = Conv1D(1024, 1, activation = 'relu')(g)
    g = BatchNormalization()(g)

    # Global Features
    global_feature = MaxPool1D(pool_size = NUM_POINTS)(g)

    # Classifier Net
    c = Dense(512, activation = 'relu')(global_feature)
    c = BatchNormalization()(c)
    c = Dropout(rate = 0.7)(c)
    c = Dense(256, activation = 'relu')(c)
    c = BatchNormalization()(c)
    c = Dropout(rate = 0.7)(c)
    c = Dense(k, activation = 'softmax')(c)

    prediction = Flatten()(c)

    model = Model(inputs = input_points, outputs = prediction)

    print("[INFO] Model Summary ---")
    print(model.summary())