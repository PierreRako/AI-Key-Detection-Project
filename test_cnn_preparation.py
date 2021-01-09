import numpy as np
import matplotlib.pyplot as plt
from prepare_CNN_data import prepare_cnn_data
import librosa, librosa.display


dataset_path = "./test_dataset/audio"
annotations_path = "./test_dataset/annotations"
names, X, keys, y = prepare_cnn_data(dataset_path,annotations_path)
print("dataset shape : ", X.shape)
print("labels shape : ", y.shape)
print("first 3 labels :\n (coded) ", y[:3], "\n (raw) ", keys[:3] )
librosa.display.specshow(X[0,:,:])
plt.show()
