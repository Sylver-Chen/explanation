from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, mean_absolute_error, explained_variance_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from keras.models import model_from_json

from csv_logger import CSVLogger
from lime_tabular import LimeTabularExplainer
from models import NN_with_EntityEmbedding, data_preprocessing, get_encoder_and_decoder, predict_proba
from data_script import get_data

filtered_tasks_ids = [3, 20, 24, 41, 45, 49, 3492, 3493, 3494, 3560, 34537, 34539, 146195]
# j is the index of filtered_tasks_ids
j = 0

# log_file = '/content/drive/My Drive/myenv/logs/coefs_stability_'+str(j)
# model_path = '/content/drive/My Drive/myenv/logs/'
log_file = 'logs/coefs_stability_'+str(j)
model_path = 'trained_models/'
logger = CSVLogger(name='mylogger', log_file=log_file, level='info')

epochs = 20
batch_size = 128
num_hidden = 500
explain_index = 0
repeat_times = 10

num_samples = 500

# Experiment 2: coefficients stability
taskid = filtered_tasks_ids[j]
# the trained model is trained with random_state=0, so fix random_state
X, X_train, X_test, y_train_str, y_test_str, y_train_int, y_test_int, feature_names, class_names, categorical_names, categorical_features = get_data(taskid, random_state=0)

# dict_with_embedding, dict_without_embedding = {}, {}
# for i in categorical_features:
#     dict_with_embedding[i] = []
#     dict_without_embedding[i] = []

# load model
with open(model_path+'model_architecture_'+str(j)+'.json', 'r') as f:
    model = model_from_json(f.read())

model.load_weights(model_path+'model_weight_'+str(j)+'.h5')
predict_fn = lambda x: predict_proba(model, x)

# (1) post explaination with nn embedding
# encoder and decoder
strategy = 'with embedding'
encoder_weight_path = 'logs/encoder_model_weight.h5'
encoder_architecture_path = 'logs/encoder_model_architecture.json'
decoder_weight_path = 'logs/decoder_model_weight.h5'
decoder_architecture_path = 'logs/decoder_model_architecture.json'

# encoder, decoder = get_encoder_and_decoder(model, X_train, num_hidden=num_hidden, epochs=epochs, batch_size=batch_size)

# # Save the weights
# encoder.save_weights(encoder_weight_path)
# decoder.save_weights(decoder_weight_path)
# # Save the model architecture
# with open(encoder_architecture_path, 'w') as f:
#     f.write(encoder.to_json())
# with open(decoder_architecture_path, 'w') as f:
#     f.write(decoder.to_json())

# load encoder and decoder
with open(encoder_architecture_path, 'r') as f:
    encoder = model_from_json(f.read())
encoder.load_weights(encoder_weight_path)
with open(decoder_architecture_path, 'r') as f:
    decoder = model_from_json(f.read())
decoder.load_weights(decoder_weight_path)

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

for repeatid in range(1, repeat_times+1):
    exp = explainer.explain_instance(X_test[explain_index],
                                     predict_fn,
                                     # num_features=2,
                                     num_samples=num_samples,
                                     top_labels=1,
    )
    # get coefs, list of tuples (feature_id, coef),
    # 按系数的绝对值排序，从大到小。
    coefs = exp.local_exp[0]
    for featureid, coef in coefs:
        # dict_with_embedding[feature_id].append(coef)
        logger.log_coef_stability(strategy, taskid, repeatid, num_samples, featureid, coef)

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
for repeatid in range(1, repeat_times+1):
    exp = explainer.explain_instance(X_test[explain_index],
                                     predict_fn,
                                     num_features=2,
                                     num_samples=num_samples,
                                     top_labels=1,
    )
    # get coefs, list of tuples (feature_id, coef),
    # 按系数的绝对值排序，从大到小。
    coefs = exp.local_exp[0]
    for featureid, coef in coefs:
        # dict_with_embedding[feature_id].append(coef)
        logger.log_coef_stability(strategy, taskid, repeatid, num_samples, featureid, coef)
