from sklearn.neural_network import MLPClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
iris = datasets.load_iris()
epochs=150
learning_rate=0.04
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)
clf = MLPClassifier(hidden_layer_sizes=(10),random_state=1,learning_rate_init=learning_rate, max_iter=epochs).fit(X_train, y_train)
array1 = clf.predict_proba(X_test[:1])
array2 = clf.predict(X_test[:5, :])
array3 = clf.score(X_test, y_test)