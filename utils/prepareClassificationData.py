# Importing necessary pakcages and libraries
import numpy as np 
import h5py
import os 

def read_off(fileName):
    
    num_select = 1024
    f = open(fileName)

    # Ignoring the 'OFF' on the first line and the second line
    f.readline() 
    f.readline()

    All_points = []
    selected_points = []

    while True:
        new_line = f.readline()
        x =  new_line.split(' ')

        if x[0] != '3':
            A = np.array(x[0:3], dtype = 'float32')
            All_points.append(A)

        else:
            break
    
    # If the number of points are less than 2000, extent the point set
    if len(All_points) < (num_select * 3):
        return None 

    # shuffle points 
    index = np.random.choice(len(All_points), num_select, replace = False)

    for i in range(len(index)):
        selected_points.append(All_points[index[i]])
    
    # return N * 3 array
    return selected_points

def save_h5(h5_filename, data, label, data_dtype = 'float32', label_dtype = 'float32'):
    
    h5_fout = h5py.File(h5_filename)
    
    h5_fout.create_dataset(
        'data', 
        data = data, 
        compression = 'gzip', 
        compression_opts = 1, 
        dtype = data_dtype)
    
    h5_fout.create_dataset(
        'label',
        data = label,
        compression = 'gzip', 
        compression_opts = 1,
        dtype = label_dtype)

    h5_fout.close()

curPath = os.path.dirname(os.path.realpath(__file__))
dirPath = os.path.join(curPath, "ModelNet40")

# List of all the categories
directories = [ d for d in os.listdir(dirPath) if os.path.isdir(os.path.join(dirPath, d))]

# Loading The Data
load_dict = [["train", "PrepTrainData"], ["test", "PrepTestData"]]

for d in load_dict:
    for i in range(len(directories)):
        label = directories[i]
        
        trainPath = os.path.join(dirPath, directories[i], d[0])
        savePath = os.path.join(curPath, d[1])

        All_points = None
        label = []

        # All the files in the "train" Folder
        for filename in os.listdir(trainPath):
            print("Loading File -", filename)

            if '.off' in filename:
                s = os.path.join(trainPath, filename)
                points = read_off(s)

                if All_points is None:
                    if points:
                        All_points = points
                        label.append(i)
                
                    elif points:
                        All_points = np.vstack((All_points, points))
                        label.append(i)
        
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        
        dataSavePath = os.path.join(savePath, directories[i] + '.h5')
        save_h5(dataSavePath, All_points, label)

        print(All_points.shape)
        print(len(label))