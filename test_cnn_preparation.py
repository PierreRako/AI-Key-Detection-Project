import numpy as np
import matplotlib.pyplot as plt
from prepare_CNN_data import prepare_cnn_data


dataset_path = "./Datasets/giantsteps-key-dataset-master/audio"
annotations_path = "./Datasets/giantsteps-key-dataset-master/annotations/key"
names, X, keys, y = prepare_cnn_data(dataset_path,annotations_path)
print(X.shape)