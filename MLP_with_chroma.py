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

def network_training(learning_rate):
    print("data loaded")
    print(chroma_df.head(3))
    X, y = np.stack(chroma_df['chromagram']), np.array(chroma_df['coded_key'].astype('int'))
    print("shape of X and y : ", X.shape,y.shape)
    #print("first row of X : ", X[0,:20],y[0])
    X_train, X_test, y_train, y_test = train_test_split(X ,y ,test_size=0.2)
    print("data separated")
    # Neural network parameters
    epochs = 400
    iter_for_convergence = epochs
    hidden_layer_sizes = (100,50,30) # these represents the number of neurons IN THE HIDDEN LAYERS ONLYS
    activation = 'logistic'
    validation= True;
    verbose=True;

    # Regularization constant

    alpha=0.0

    # NNet init
    clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, 
    random_state=1, max_iter=epochs, learning_rate_init=learning_rate,learning_rate='adaptive',
    alpha=alpha, early_stopping=validation, verbose=verbose, n_iter_no_change=iter_for_convergence)
    print("Classifier initialized")
    # NNet training
    print("fitting starts...")
    clf.fit(X_train,y_train)
    print("fitting finished")
    # Some insights on the training
    array1 = clf.predict_proba(X_test[:1])
    array2 = clf.predict(X_test[:5, :])
    array3 = clf.score(X_test, y_test)
    print("look for variables")
    return(clf.loss_,clf,X_test,X_train,y_test,y_train)

loss,clf,X_te,X_tr,y_te,y_tr = network_training(0.008)
print(loss)

def search_learning_rate():
    losses= []
    ii = np.linspace(10**-5,6*10**-2,100)

    for i in range(0,100):
        losses += [network_training(ii[i])[0]]
    plt.plot(ii,losses)
    plt.show()