from sklearn.model_selection import cross_val_score, KFold, validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

def KFold_data(K, model, x_data, y_data):
    cv = KFold(K, shuffle=True, random_state=0)
    scores = cross_val_score(model, x_data, y_data, cv=cv)
    return scores

# Using Cross validation to find the best parameters for a given model
# @param model
# @param_dict dictionnary of all the parameters to optimize
# @param n_folds number of folds
# @param x_train and y_train are the training data
# @return grid.best_params_ values of best parameters
# @return grid.best_estimator best model, use it with .score(x_test, y_test) to evaluate test performance
def optimize_param(model, param_dict, n_folds, x_train, y_train):
    grid = GridSearchCV(model, param_dict, cv=n_folds)
    grid.fit(x_train, y_train)
    return grid.best_params_, grid.best_estimator_

def display_results_matrix(model, x_test, y_test):
    #Make sure to have the predict method for the model
    return confusion_matrix(y_test, model.predict(x_test))