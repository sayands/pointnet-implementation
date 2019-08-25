# Importing keras and tensorflow related packages
from keras import optimizers
from keras.layers import * 
from keras.models import * 

def createClassModel(model, NUM_POINTS, k):
    # Global Features
    global_feature = MaxPool1D(pool_size = NUM_POINTS)(model.output)

    # Classifier Net
    c = Dense(512, activation = 'relu', kernel_initializer='he_normal')(global_feature)
    c = BatchNormalization()(c)
    c = Dropout(rate = 0.7)(c)
    c = Dense(256, activation = 'relu', kernel_initializer='he_normal')(c)
    c = BatchNormalization()(c)
    c = Dropout(rate = 0.7)(c)

    # Output Classification Layer
    c = Dense(k, activation = 'softmax', kernel_initializer='he_normal')(c)
    prediction = Flatten()(c)

    # Creating Model
    model = Model(inputs = model.input, outputs = prediction)

    return model