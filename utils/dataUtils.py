# Importing neccessary packages and libraries
import tensorflow as tf 
import  h5py
import numpy as np
def mat_mul(A, B):
    return tf.matmul(A, B)

def rotate_point_cloud(batch_data):
    '''
    Randomly rotate the point cloud to augment the dataset, rotation is per
    shape based along up direction

    Input : B x N x 3 array, original batch of point clouds
    Returns : B x N x 3 array, rotated batch of point clouds
    '''

    rotated_data = np.zeros(batch_data.shape, dtype = np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi 
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)

        rotation_matrix = np.array([[cosval, 0, sinval],
                                                [0, 1, 0],
                                                [-sinval, 0, cosval]])
        
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    
    return rotated_data

def jitter_point_cloud(batch_data, sigma = 0.01, clip = 0.05):
    '''
    Randomly jitter points, jittering is per point. 

    Input : B x N x 3 array, original batch of point clouds
    Returns : B x N x 3 array, jittered batch of point clouds
    '''

    B, N, C = batch_data.shape
    asser(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
    jittered_data += batch_data
    return jittered_data
