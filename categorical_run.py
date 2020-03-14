import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import OneHotEncoder
import openml
from sklearn.datasets import fetch_openml

# from lime.lime_tabular import LimeTabularExplainer
from lime_tabular import LimeTabularExplainer
from models import NN_with_EntityEmbedding

# loading data
def get_data():
    task = openml.tasks.get_task(3)
    X, y = fetch_openml(data_id=task.dataset_id, as_frame=True, return_X_y=True)
    feature_names = X.columns.tolist()
    # class_names = y.cat.categories.tolist()

    # dataframe to numpy
    X = X.to_numpy()
    y = y.to_numpy()

    # samll size for test
    X = X[2200:2500]
    y = y[2200:2500]

    categorical_features = range(np.shape(X)[1])

    categorical_names = {}
    for feature in categorical_features:
        le = LabelEncoder()
        le.fit(X[:, feature])
        X[:, feature] = le.transform(X[:, feature])
        categorical_names[feature] = le.classes_


    le = LabelEncoder()
    le.fit(y[:])
    y[:] = le.transform(y[:])
    class_names = le.classes_
    # split the data into training and testing
    X = X.astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=0)
    return X, X_train, X_test, y_train, y_test, feature_names, class_names, categorical_names, categorical_features

if __name__ == "__main__":
    X, X_train, X_test, y_train, y_test, feature_names, class_names, categorical_names, categorical_features = get_data()

    # encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    # encoder.fit(X)
    # encoded_X_train = encoder.transform(X_train)

    # rf = RandomForestClassifier(n_estimators=50)
    # rf.fit(encoded_X_train, y_train)
    # sklearn.metrics.accuracy_score(labels_test, rf.predict(test))
    # predict_fn = lambda x: rf.predict_proba(encoder.transform(x))

    nn = NN_with_EntityEmbedding(X_train, y_train,
                                 categorical_features, categorical_names,
                                 class_names,
                                 epochs=1,
                                 batch_size=160,

    )
    # to do: test NN performance
    score = nn.evaluate(X_test, y_test)
    print("nn prediction score: ", str(score))

    predict_fn = lambda x: nn.predict_proba(x)

    explainer = LimeTabularExplainer(X_train,
                                     class_names=class_names,
                                     feature_names=feature_names,
                                     categorical_features=categorical_features,
                                     categorical_names=categorical_names,
                                     kernel_width=3,
                                     verbose=False,
                                     sample_around_instance=True,
                                     # discretize_continuous=False,
    )

    # Experiment 1
    # right means the prediction made by blackbox
    right_prob = []
    # fit means the prediction made by linear model
    fit_prob = []
    print("testing set size:  ", X_test.shape[0])
    for i in range(X_test.shape[0]):
        print(str(i+1)+"th test instance")
        exp = explainer.explain_instance(X_test[i],
                                         predict_fn,
                                         # num_features=2,
                                         num_samples=50,
                                         top_labels=1)
        right_prob.append(exp.right)
        fit_prob.append(exp.local_pred)

    # calculate results
    score_r2 = r2_score(right_prob, fit_prob)
    score_mse = mean_squared_error(right_prob, fit_prob)
    print("r2 score: ", score_r2)
    print("mse sore: ", score_mse)
