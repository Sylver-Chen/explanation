import numpy as np
import pandas as pd
import copy
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import OneHotEncoder
import openml
from sklearn.datasets import fetch_openml

# from lime.lime_tabular import LimeTabularExplainer
from lime_tabular import LimeTabularExplainer
from models import NN_with_EntityEmbedding, NN
from csv_logger import CSVLogger
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

    # # samll size for test
    # X = X[2200:2500]
    # y = y[2200:2500]
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



if __name__ == "__main__":
    log_file = './logs/log'
    logger = CSVLogger(name='mylogger', log_file=log_file, level='info')
    filtered_tasks_ids = [3, 20, 24, 41, 45, 49, 3492, 3493, 3494, 3560, 34537, 34539, 146195]
    epochs = 50
    batch_size = 128
    for i in range(5):
        for j in range(len(filtered_tasks_ids)):
            # within different runs, split traning/testing sets with different
            # random_state.
            taskid = filtered_tasks_ids[j]
            X, X_train, X_test, y_train_str, y_test_str, y_train_int, y_test_int, feature_names, class_names, categorical_names, categorical_features = get_data(taskid, random_state=i)

            # nn with embedding related
            nn_with_embedding = NN_with_EntityEmbedding(X_train, y_train_int,
                                                        categorical_features,
                                                        categorical_names,
                                                        class_names,
                                                        epochs=epochs,
                                                        batch_size=batch_size,
            )
            nn_with_embedding_score = nn_with_embedding.evaluate(X_test, y_test_int)
            print("nn_with_embedding prediction score: ", str(nn_with_embedding_score))
            logger.log('nn_with_embedding', taskid, i+1, nn_with_embedding_score)

            # nn related
            nn = NN(X_train,
                    y_train_int,
                    categorical_features,
                    categorical_names,
                    class_names,
                    epochs=epochs,
                    batch_size=batch_size,)
            nn_score = nn.evaluate(X_test, y_test_int)
            print("nn prediction score: ", str(nn_score))
            logger.log('nn', taskid, i+1, nn_score)

            # machine learning model related
            encoder_X = OneHotEncoder(handle_unknown='ignore', sparse=True)
            encoder_X.fit(X)
            encoded_X_train = encoder_X.transform(X_train)
            encoded_X_test = encoder_X.transform(X_test)

            rf = RandomForestClassifier(n_estimators=100)
            rf.fit(encoded_X_train, y_train_str)
            # predict_fn = lambda x: rf.predict_proba(encoder.transform(x))
            rf_score = accuracy_score(rf.predict(encoded_X_test), y_test_str)
            print("random forest score: ", str(rf_score))
            logger.log('random_forest', taskid, i+1, rf_score)

    # predict_fn = lambda x: nn.predict_proba(x)

    # explainer = LimeTabularExplainer(X_train,
    #                                  class_names=class_names,
    #                                  feature_names=feature_names,
    #                                  categorical_features=categorical_features,
    #                                  categorical_names=categorical_names,
    #                                  kernel_width=3,
    #                                  verbose=False,
    #                                  sample_around_instance=True,
    #                                  # discretize_continuous=False,
    # )

    # # Experiment 1
    # # right means the prediction made by blackbox
    # right_prob = []
    # # fit means the prediction made by linear model
    # fit_prob = []
    # print("testing set size:  ", X_test.shape[0])
    # for i in range(X_test.shape[0]):
    #     print(str(i+1)+"th test instance")
    #     exp = explainer.explain_instance(X_test[i],
    #                                      predict_fn,
    #                                      # num_features=2,
    #                                      num_samples=50,
    #                                      top_labels=1)
    #     right_prob.append(exp.right)
    #     fit_prob.append(exp.local_pred)

    # # calculate results
    # score_r2 = r2_score(right_prob, fit_prob)
    # score_mse = mean_squared_error(right_prob, fit_prob)
    # print("r2 score: ", score_r2)
    # print("mse sore: ", score_mse)
