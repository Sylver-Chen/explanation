import numpy as np
import pandas as pd
import copy
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import openml
from sklearn.datasets import fetch_openml


# loading data
def get_data(taskid, random_state=0):
    task = openml.tasks.get_task(taskid)
    print(task)
    X, y = fetch_openml(data_id=task.dataset_id, as_frame=True, return_X_y=True)
    feature_names = X.columns.tolist()
    # class_names = y.cat.categories.tolist()

    # dataframe to numpy
    X = X.to_numpy()
    y = y.to_numpy()

    # samll size for test
    X = X[2200:2500]
    y = y[2200:2500]
    print("sample size: ", X.shape)

    categorical_features = range(np.shape(X)[1])

    categorical_names = {}
    for feature in categorical_features:
        le = LabelEncoder()
        col = X[:, feature]
        # todo: optimization
        for idx, el in enumerate(col):
            if pd.isnull(el):
                col[idx] = 'nan'

        le.fit(col)
        X[:, feature] = le.transform(col)
        categorical_names[feature] = le.classes_
        # le.fit(X[:, feature])
        # X[:, feature] = le.transform(X[:, feature])
        # categorical_names[feature] = le.classes_

    # split the data into training and testing
    # sklearn model use string labels
    X = X.astype(float)
    X_train, X_test, y_train_str, y_test_str = train_test_split(X, y, test_size=0.3, random_state=random_state)
    y_str = y
    # keras model use integer labels
    y_int = copy.deepcopy(y_str)
    y_train_int = copy.deepcopy(y_train_str)
    y_test_int = copy.deepcopy(y_test_str)

    le = LabelEncoder()
    le.fit(y_int[:])
    y_int[:] = le.transform(y_int[:])
    class_names = le.classes_
    y_train_int[:] = le.transform(y_train_int[:])
    y_test_int[:] = le.transform(y_test_int[:])

    return X, X_train, X_test, y_train_str, y_test_str, y_train_int, y_test_int, feature_names, class_names, categorical_names, categorical_features
