import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
from keras import layers
import matplotlib.pyplot as plt
import librosa, librosa.display
from sklearn.model_selection import train_test_split

verbose = False

np_input_path= "./numpy_material/GS_key_dataset_inputs_preprocessed.npy"
np_label_path= "./numpy_material/GS_key_dataset_labels_preprocessed.npy"
X = np.load(np_input_path)
y = np.load(np_label_path)

if verbose:
    print("y[0] : ", y[0])
    librosa.display.specshow(X[0,:,:])
    plt.show()

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

X_train = X_train.reshape(-1,120,100,1)
X_test = X_test.reshape(-1,120,100,1)

model = keras.Sequential()
model.add(keras.Input(shape = (120,100,1)))
for _ in range(5):
    model.add(layers.Conv2D(
        8,
        kernel_size=5,
        activation="relu",
        padding="same"
        ))

model.summary()

model.add(layers.Permute((2,1,3)))

model.summary()

model.add(layers.Reshape((100,-1)))

model.add(layers.Dense(48,activation="relu"))

model.summary()

model.add(layers.GlobalAveragePooling1D())

model.summary()

model.add(layers.Dense(24,activation="softmax"))

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])