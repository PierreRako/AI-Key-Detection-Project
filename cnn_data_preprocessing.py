import numpy as np
from prepare_CNN_data import prepare_cnn_data

# THIS SCRIPT CREATES THE NP ARRAYS WE WILL USE IN THE CNN AND SAVE THEM FOR LATER USE IN .npy FILES

dataset_path = "./Datasets/giantsteps-key-dataset-master/audio"
annotations_path =  "./Datasets/giantsteps-key-dataset-master/annotations/key"
names, X, keys, y = prepare_cnn_data(dataset_path,annotations_path)

outpath_X = "./numpy_material/GS_key_dataset_inputs_preprocessed"
outpath_Y = "./numpy_material/GS_key_dataset_labels_preprocessed"

np.save(outpath_X,X)
np.save(outpath_Y,y)