import numpy as np
import sklearn
import  pandas as pd
import copy
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
import openml
from sklearn.datasets import fetch_openml
import seaborn as sns
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.models import Model as KerasModel
from keras.layers import Input, Dense, Activation, Reshape
from keras.layers import Concatenate
from keras.layers.embeddings import Embedding
from keras.models import model_from_json


from lime_tabular import LimeTabularExplainer
from models import NN_with_EntityEmbedding


filtered_tasks_ids = [3, 20, 24, 41, 45, 49, 3492, 3493, 3494, 3560, 34537, 34539, 146195]
taskid = filtered_tasks_ids[0]
X, X_train, X_test, y_train_str, y_test_str, y_train_int, y_test_int, feature_names, class_names, categorical_names, categorical_features = get_data(taskid, random_state=0)

nn_with_embedding = NN_with_EntityEmbedding(X_train, y_train_int,
                                            categorical_features,
                                            categorical_names,
                                            class_names,
                                            epochs=1,
                                            batch_size=128,
)

model = nn_with_embedding.model
with open('logs/nn_with_embedding_model_architecture.json', 'r') as f:
    model = model_from_json(f.read())
model.load_weights('logs/nn_with_embedding_model_weights.h5')

layer_name = 'fully-connected'

previous_layer_name = 'embedding'
intermediate_layer_model = KerasModel(inputs=model.input,
                                      outputs=model.get_layer(previous_layer_name).output)
intermediate_output = intermediate_layer_model.predict(preprocessing(X_test))
layer_names = [layer.name for layer in model.layers]
layer_idx = layer_names.index(layer_name)
# get the input shape of the desired layer
input_shape = model.layers[layer_idx].get_input_shape_at(0)
new_input = Input(shape=(input_shape[1],))
# create the new nodes for each layer in the path
x = new_input
for layer in model.layers[layer_idx:]:
    x = layer(x)

new_model = KerasModel(new_input, x)
a = new_model.predict(intermediate_output)
