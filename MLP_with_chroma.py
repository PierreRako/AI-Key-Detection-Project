import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import pandas as pd

# Dataset loading (panda) and separation



# Neural network parameters

epochs = 100
learning_rate = 0.01
hidden_layer_sizes = (100,50,30)
activation = 'logistic'

clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,activation=activation,
random_state=1,max_iter=epochs,learning_rate=learning_rate)

