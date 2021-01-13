import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
from keras import layers
import matplotlib.pyplot as plt
import librosa, librosa.display
from sklearn.model_selection import train_test_split

verbose = True

np_input_path= "./numpy_material/GS-mtg_key_dataset_inputs_preprocessed.npy"
np_label_path= "./numpy_material/GS-mtg_key_dataset_labels_preprocessed.npy"
X = np.load(np_input_path)
y = np.load(np_label_path)

if verbose:
    print("y[0] : ", y[0])
    librosa.display.specshow(X[0,:,:])
    print(X.shape)
    print(y.shape)
    #plt.show()

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

X_train = X_train.reshape(-1,120,100,1)
X_test = X_test.reshape(-1,120,100,1)

if verbose: print(X_train.shape, "\n", y_train.shape, "\n")
if verbose: print("number of validation samples : ", X_test.shape)

model = keras.Sequential()
model.add(layers.Conv2D(
        8,
        kernel_size=5,
        activation="relu",
        padding="same",
        input_shape=(120,100,1)
        ))
for _ in range(4):
    model.add(layers.Conv2D(
        8,
        kernel_size=5,
        activation="relu",
        padding="same",
        kernel_regularizer=keras.regularizers.l2(l2=0.0001)
        ))

model.add(layers.Permute((2,1,3)))

model.add(layers.Reshape((100,-1)))

model.add(layers.Dense(48,activation="relu",kernel_regularizer=keras.regularizers.l2(l2=0.0001)))

model.add(layers.GlobalAveragePooling1D())

#if verbose: model.summary()

model.add(layers.Dense(24,activation="softmax"))

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=100, batch_size=64)

print(finished)