from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, mean_absolute_error, explained_variance_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from keras.models import model_from_json

from csv_logger import CSVLogger
from lime_tabular import LimeTabularExplainer
from models import NN_with_EntityEmbedding, data_preprocessing, get_encoder_and_decoder, predict_proba
from data_script import get_data

runid = 1
filtered_tasks_ids = [3, 20, 24, 41, 45, 49, 3492, 3493, 3494, 3560, 34537, 34539, 146195]
# filtered_tasks_ids = [3]
# j is the index of filtered_tasks_ids
j = 0
log_file = '/content/drive/My Drive/myenv/logs/local_fidelity_'+str(j)
model_path = '/content/drive/My Drive/myenv/trained_models/'
# log_file = 'logs/local_fidelity_'+str(j)
# model_path = 'trained_models/'
logger = CSVLogger(name='mylogger', log_file=log_file, level='info')
num_samples_iter = list(range(1000, 5500, 500))
epochs = 20
batch_size = 128
num_hidden = 500

strategy_list, taskid_list, runid_list, num_samples_list, score_r2_list, score_mse_list, score_mae_list, score_evc_list = [], [], [], [], [], [], [], []

# Experiment 1: local fidelity
for num_samples in num_samples_iter:
    print("++++++++++++++++++++++++++++++++++++++++++++++++")
    print("+++ num_samples: ", num_samples)
    print("++++++++++++++++++++++++++++++++++++++++++++++++")

    taskid = filtered_tasks_ids[j]
    # the trained model is trained with random_state=0, so fix random_state
    X, X_train, X_test, y_train_str, y_test_str, y_train_int, y_test_int, feature_names, class_names, categorical_names, categorical_features = get_data(taskid, random_state=0)
    print("===== testing set size:  ", X_test.shape[0])

    # load model
    with open(model_path+'model_architecture_'+str(j)+'.json', 'r') as f:
        model = model_from_json(f.read())

    model.load_weights(model_path+'model_weight_'+str(j)+'.h5')
    predict_fn = lambda x: predict_proba(model, x)

    # (1) post explaination with nn embedding
    # encoder and decoder
    strategy = 'with embedding'
    # encoder_weight_path = 'logs/encoder_model_weight.h5'
    # encoder_architecture_path = 'logs/encoder_model_architecture.json'
    # decoder_weight_path = 'logs/decoder_model_weight.h5'
    # decoder_architecture_path = 'logs/decoder_model_architecture.json'

    encoder, decoder = get_encoder_and_decoder(model, X_train, num_hidden=num_hidden, epochs=epochs, batch_size=batch_size)
    # # Save the weights
    # encoder.save_weights(encoder_weight_path)
    # decoder.save_weights(decoder_weight_path)
    # # Save the model architecture
    # with open(encoder_architecture_path, 'w') as f:
    #     f.write(encoder.to_json())
    # with open(decoder_architecture_path, 'w') as f:
    #     f.write(decoder.to_json())

    # # load encoder and decoder
    # with open(encoder_architecture_path, 'r') as f:
    #     encoder = model_from_json(f.read())
    # encoder.load_weights(encoder_weight_path)
    # with open(decoder_architecture_path, 'r') as f:
    #     decoder = model_from_json(f.read())
    # decoder.load_weights(decoder_weight_path)

    explainer = LimeTabularExplainer(X_train,
                                     class_names=class_names,
                                     feature_names=feature_names,
                                     categorical_features=categorical_features,
                                     categorical_names=categorical_names,
                                     # kernel_width=3,
                                     verbose=False,
                                     sample_around_instance=True,
                                     encoder=encoder,
                                     decoder=decoder,
                                     # discretize_continuous=False,
    )

    print("===== strategy: ", strategy)
    # different number of sample points 
    # right means the prediction made by blackbox
    right_prob = []
    # fit means the prediction made by linear model
    fit_prob = []
    for i in range(X_test.shape[0]):
        # print(str(i+1)+"th test instance")
        # change num_samples
        exp = explainer.explain_instance(X_test[i],
                                         predict_fn,
                                         # num_features=2,
                                         num_samples=num_samples,
                                         top_labels=1,
        )
        right_prob.append(exp.right)
        fit_prob.append(exp.local_pred)

    # calculate results
    score_r2 = r2_score(right_prob, fit_prob)
    score_mse = mean_squared_error(right_prob, fit_prob)
    score_mae = mean_absolute_error(right_prob, fit_prob)
    score_evc = explained_variance_score(right_prob, fit_prob)

    print('R2: ', score_r2)
    print('Mean Squared Error: ', score_mse)
    print('Mean Absolute Error: ', score_mae)
    print('Explain Variance Score: ', score_evc)
    logger.log_local_fidelity(strategy, taskid, runid, num_samples, score_r2, score_mse, score_mae, score_evc)
    strategy_list.append(strategy)
    taskid_list.append(taskid)
    runid_list.append(runid)
    num_samples_list.append(num_samples)
    score_r2_list.append(score_r2)
    score_mse_list.append(score_mse)
    score_mae_list.append(score_mae)
    score_evc_list.append(score_evc)

    # (2) post explaination without nn embedding
    strategy = 'without embedding'
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

    print("===== strategy: ", strategy)
    # different number of sample points
    # right means the prediction made by blackbox
    right_prob = []
    # fit means the prediction made by linear model
    fit_prob = []
    # print("testing set size:  ", X_test.shape[0])
    for i in range(X_test.shape[0]):
        # print(str(i+1)+"th test instance")
        # change num_samples
        exp = explainer.explain_instance(X_test[i],
                                         predict_fn,
                                         num_features=2,
                                         num_samples=num_samples,
                                         top_labels=1,
        )
        right_prob.append(exp.right)
        fit_prob.append(exp.local_pred)

    # calculate results
    score_r2 = r2_score(right_prob, fit_prob)
    score_mse = mean_squared_error(right_prob, fit_prob)
    score_mae = mean_absolute_error(right_prob, fit_prob)
    score_evc = explained_variance_score(right_prob, fit_prob)

    print('R2: ', score_r2)
    print('Mean Squared Error: ', score_mse)
    print('Mean Absolute Error: ', score_mae)
    print('Explain Variance Score: ', score_evc)
    logger.log_local_fidelity(strategy, taskid, runid, num_samples, score_r2, score_mse, score_mae, score_evc)
    strategy_list.append(strategy)
    taskid_list.append(taskid)
    runid_list.append(runid)
    num_samples_list.append(num_samples)
    score_r2_list.append(score_r2)
    score_mse_list.append(score_mse)
    score_mae_list.append(score_mae)
    score_evc_list.append(score_evc)


print("num_samples: ", num_samples)
print("strategy_list: ", strategy_list)
print("taskid_list: ", taskid_list)
print("runid_list: ", runid_list)
print("num_samples_list: ", num_samples_list)
print("score_r2_list: ", score_r2_list)
print("score_mse_list: ", score_mse_list)
print("score_mae_list: ", score_mae_list)
print("score_evc_list: ", score_evc_list)