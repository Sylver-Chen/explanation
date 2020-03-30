from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, mean_absolute_error, explained_variance_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from keras.models import model_from_json

from csv_logger import CSVLogger
from lime_tabular import LimeTabularExplainer
from models import NN_with_EntityEmbedding, data_preprocessing, get_encoder_and_decoder, get_autoencoder, predict_proba
from data_script import get_data

filtered_tasks_ids = [3, 20, 24, 41, 45, 49, 3492, 3493, 3494, 3560, 34537, 34539, 146195]
# j is the index of filtered_tasks_ids
j = 0

# Google Colab
log_file = '/content/drive/My Drive/myenv/logs/ae_mlp_coefs_stability_'+str(j)
blackbox_path = '/content/drive/My Drive/myenv/trained_models_with_entity_embedding/'
embedding_path = '/content/drive/My Drive/myenv/trained_models_with_mlp_embedding/'

# # Local Host
# log_file = 'logs/try_mlp_embedding_local_fidelity_'+str(j)
# blackbox_path = 'trained_models_with_entity_embedding/'
# embedding_path = 'trained_models_with_mlp_embedding/'


logger = CSVLogger(name='mylogger', log_file=log_file, level='info')
num_samples_iter = list(range(500, 5500, 500))

epochs = 50
# epochs = 1
batch_size = 128
num_hidden = 500
# num_hidden = 10
explain_index = 0
repeat_times = 10
# repeat_times = 1

# change ! 
# blackbox = 'rf'
blackbox = 'nn' 
embedding_type = 'mlp' # 'entity'
# embedding_type = 'entity'

# Experiment 2: coefficients stability
taskid = filtered_tasks_ids[j]
# the trained model is trained with random_state=0, so fix random_state
X, X_train, X_test, y_train_str, y_test_str, y_train_int, y_test_int, feature_names, class_names, categorical_names, categorical_features = get_data(taskid, random_state=0)
# load blackbox
with open(blackbox_path+'model_architecture_'+str(j)+'.json', 'r') as f:
    blackbox_model = model_from_json(f.read())

blackbox_model.load_weights(blackbox_path+'model_weight_'+str(j)+'.h5')

# # blackbox model:
if blackbox == 'nn':
    # (1) nueral network
    predict_fn = lambda x: predict_proba(blackbox_model, x)
else:
    # (2) random forest
    encoder_X = OneHotEncoder(handle_unknown='ignore', sparse=True)
    encoder_X.fit(X)
    encoded_X_train = encoder_X.transform(X_train)
    encoded_X_test = encoder_X.transform(X_test)

    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(encoded_X_train, y_train_str)
    predict_fn = lambda x: rf.predict_proba(encoder_X.transform(x))


# (1) post explaination with nn embedding
# encoder and decoder
# strategy = 'with embedding'
# strategy = 'with mlp embedding'
strategy = 'with ae mlp embedding'

# with open(embedding_path+'model_architecture_'+str(j)+'.json', 'r') as f:
#     embedding_model = model_from_json(f.read())

# embedding_model.load_weights(embedding_path+'model_weight_'+str(j)+'.h5')

# # embedding_model = blackbox_model
# encoder, decoder = get_encoder_and_decoder(embedding_model,
#                                            X_train, num_hidden=num_hidden,
#                                            embedding_type=embedding_type,
#                                            epochs=epochs, batch_size=batch_size, )

encoder, decoder = get_autoencoder(categorical_features, categorical_names,
                                   X_train, num_hidden=num_hidden,
                                   embedding_type=embedding_type,
                                   epochs=epochs, batch_size=batch_size,
)

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
                                 embedding_type=embedding_type,
                                 # discretize_continuous=False,
                                 feature_selection='auto', # 'auto', # 'none'
)

print("===== strategy: ", strategy)

for num_samples in num_samples_iter:
    for repeatid in range(1, repeat_times+1):
        exp = explainer.explain_instance(X_test[explain_index],
                                         predict_fn,
                                         # num_features=2,
                                         num_samples=num_samples,
                                         top_labels=1,
        )
        # get coefs, list of tuples (feature_id, coef),
        # 按系数的绝对值排序，从大到小。
        label_idx = exp.top_labels[0]
        coefs = exp.local_exp[label_idx]
        selected = []
        for featureid, coef in coefs:
            selected.append(featureid)
            # dict_with_embedding[feature_id].append(coef)
            logger.log_coef_stability(strategy, taskid, repeatid, num_samples, featureid, coef)
        # 为了计算方便，将没有选择的特征进行填充，系数为0.
        for featureid in categorical_features:
            if featureid in selected:
                continue
            else:
                logger.log_coef_stability(strategy, taskid, repeatid, num_samples, featureid, 0)


# # (2) post explaination without nn embedding
# strategy = 'without embedding'
# explainer = LimeTabularExplainer(X_train,
#                                  class_names=class_names,
#                                  feature_names=feature_names,
#                                  categorical_features=categorical_features,
#                                  categorical_names=categorical_names,
#                                  # kernel_width=3,
#                                  verbose=False,
#                                  sample_around_instance=True,
#                                  # discretize_continuous=False,
#                                  feature_selection='auto', #'auto', # 'none'
# )

# print("===== strategy: ", strategy)

# for num_samples in num_samples_iter:
#     for repeatid in range(1, repeat_times+1):
#         exp = explainer.explain_instance(X_test[explain_index],
#                                          predict_fn,
#                                          # num_features=2,
#                                          num_samples=num_samples,
#                                          top_labels=1,
#         )
#         # get coefs, list of tuples (feature_id, coef),
#         # 按系数的绝对值排序，从大到小。
#         label_idx = exp.top_labels[0]
#         coefs = exp.local_exp[label_idx]
#         selected = []
#         for featureid, coef in coefs:
#             selected.append(featureid)
#             # dict_with_embedding[feature_id].append(coef)
#             logger.log_coef_stability(strategy, taskid, repeatid, num_samples, featureid, coef)
#         # 为了计算方便，将没有选择的特征进行填充，系数为0.
#         for featureid in categorical_features:
#             if featureid in selected:
#                 continue
#             else:
#                 logger.log_coef_stability(strategy, taskid, repeatid, num_samples, featureid, 0)
