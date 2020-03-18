from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

# from lime.lime_tabular import LimeTabularExplainer
from csv_logger import CSVLogger
from lime_tabular import LimeTabularExplainer
from models import NN_with_EntityEmbedding, NN
from data_script import get_data

run_id = 0
log_file = '/content/drive/My Drive/myenv/log'+str(run_id)
logger = CSVLogger(name='mylogger', log_file=log_file, level='info')
filtered_tasks_ids = [3, 20, 24, 41, 45, 49, 3492, 3493, 3494, 3560, 34537, 34539, 146195]
epochs = 50
batch_size = 128

for j in range(len(filtered_tasks_ids)):
    print("++++++++++++++++++++++++++++++++++++++++++++++++")
    print("++++++++++ tasks: "+str(j+1)+"/"+str(len(filtered_tasks_ids))+"+++")
    print("++++++++++++++++++++++++++++++++++++++++++++++++")
    # within different runs, split traning/testing sets with different
    # random_state.
    taskid = filtered_tasks_ids[j]
    X, X_train, X_test, y_train_str, y_test_str, y_train_int, y_test_int, feature_names, class_names, categorical_names, categorical_features = get_data(taskid, random_state=run_id)

    # nn with embedding related
    nn_with_embedding = NN_with_EntityEmbedding(X_train, y_train_int,
                                                categorical_features,
                                                categorical_names,
                                                class_names,
                                                epochs=epochs,
                                                batch_size=batch_size,
    )
    nn_with_embedding_loss, nn_with_embedding_score = nn_with_embedding.evaluate(X_test, y_test_int)
    print("nn_with_embedding prediction score: ", str(nn_with_embedding_score))
    logger.log('nn_with_embedding', taskid, run_id, nn_with_embedding_score)

    # nn related
    nn = NN(X_train,
            y_train_int,
            categorical_features,
            categorical_names,
            class_names,
            epochs=epochs,
            batch_size=batch_size,)
    nn_loss, nn_score = nn.evaluate(X_test, y_test_int)
    print("nn prediction score: ", str(nn_score))
    logger.log('nn', taskid, run_id, nn_score)

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
    logger.log('random_forest', taskid, run_id, rf_score)

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
