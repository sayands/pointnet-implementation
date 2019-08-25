# Importing keras and tensorflow related packages
from keras import optimizers
from keras.layers import * 
from keras.models import * 

sys.path.append('../')
from utils.dataUtils import exp_dim

def createSegModel(model, seg_part1, NUM_POINTS, k):
    # Global Features
    global_feature = MaxPool1D(pool_size = NUM_POINTS)(model.output)
    global_feature = Lambda(exp_dim, arguments = { 'num_points' : NUM_POINTS})(global_feature)

    # PointNet Segmentation Network
    c = concatenate([seg_part1, global_feature])
    c = Conv1D(512, 1, activation = 'relu', kernel_initializer='he_normal')(c)
    c = BatchNormalization()(c)
    c = Conv1D(256, 1, activation = 'relu', kernel_initializer='he_normal')(c)
    c = BatchNormalization()(c)
    c = Conv1D(128, 1, activation = 'relu', kernel_initializer='he_normal')(c)
    c = BatchNormalization()(c)
    c = Conv1D(128, 1, activation = 'relu', kernel_initializer='he_normal')(c)
    c = BatchNormalization()(c)

    prediction = Conv1D(k, 1, activation = 'softmax', kernel_initializer='he_normal')(c)
    # Creating Model
    model = Model(inputs = model.input, outputs = prediction)

    return model