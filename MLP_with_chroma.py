import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from loading_dataset import prepare_panda_dataFrame
import pandas as pd

# Dataset loading and separation
dataset_path="./Datasets/giantsteps-key-dataset-master/audio"
key_annotation_path="./Datasets/giantsteps-key-dataset-master/annotations/key"
chroma_df=prepare_panda_dataFrame(dataset_path,key_annotation_path)

X, y = chroma_df['chromagram'], chroma_df['codedkey']
X_train, X_test, y_train, y_test = train_test_split(X ,y ,test_size=0.2)

# Neural network parameters
epochs = 200
learning_rate = 0.01
hidden_layer_sizes = (100,50,30) # these represents the number of neurons IN THE HIDDEN LAYERS ONLYS
activation = 'logistic'

# NNet init
clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, 
random_state=1, max_iter=epochs, learning_rate_init=learning_rate)

# NNet training
clf.fit(X_train,y_train)

# Some insights on the training
array1 = clf.predict_proba(X_test[:1])
array2 = clf.predict(X_test[:5, :])
array3 = clf.score(X_test, y_test)