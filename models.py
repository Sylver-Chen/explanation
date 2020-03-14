import math
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn import neighbors

# import os
# os.environ['KERAS_BACKEND'] = 'theano'

from keras.models import Sequential
from keras.models import Model as KerasModel
from keras.layers import Input, Dense, Activation, Reshape
from keras.layers import Concatenate
from keras.layers.embeddings import Embedding


def split_features(X):
    X_list = []
    for i in range(X.shape[1]):
        # X_list.append(X[..., [i]])
        X_list.append(X[..., [i]])
    return X_list


class NN_with_EntityEmbedding(KerasModel):
    def __init__(self, X_train, y_train, # X_val, y_val,
                 categorical_features, categorical_names, class_names,
                 epochs=10,
                 batch_size=128,
                 mode='classification',
    ):
        super().__init__()
        self.categorical_features = categorical_features
        self.categorical_names = categorical_names
        self.class_names = class_names
        self.epochs = epochs
        self.batch_size = batch_size
        self.mode = mode
        self.__build_keras_model()
        self.fit(X_train, y_train)

    def preprocessing(self, X):
        X_list = split_features(X)
        return X_list

    def __build_keras_model(self):
        input_model, output_embeddings = [], []
        for col in self.categorical_features:
            # determine number of different values of each column
            input_dim = len(self.categorical_names[col])
            # empirical [2017, Wang]
            output_dim = 6 * int(math.pow(input_dim, 1/4))
            input = Input(shape=(1,))
            output = Embedding(input_dim, output_dim)(input)
            output = Reshape(target_shape=(output_dim, ))(output)

            input_model.append(input)
            output_embeddings.append(output)

        output_model = Concatenate()(output_embeddings)
        output_model = Dense(1000, kernel_initializer="uniform")(output_model)
        output_model = Activation('relu')(output_model)
        output_model = Dense(500, kernel_initializer="uniform")(output_model)
        output_model = Activation('relu')(output_model)
        if self.mode == 'classification':
            output_model = Dense(len(self.class_names))(output_model)
            output_model = Activation('softmax')(output_model)
        else:
            output_model = Dense(1)(output_model)
            output_model = Activation('sigmoid')(output_model)

        self.model = KerasModel(inputs=input_model, outputs=output_model)

        if self.mode == 'classification':
            self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['sparse_categorical_accuracy'])
        else:
            self.model.compile(loss='mean_absolute_error', optimizer='adam')

    def fit(self, X_train, y_train):
        self.model.fit(self.preprocessing(X_train),
                       y_train,
                       epochs=self.epochs, batch_size=self.batch_size,)

    def predict_proba(self, x):
        x = self.preprocessing(x)
        result = self.model.predict(x)#.flatten()
        return result

    def evaluate(self, X_test, y_test):
        score = self.model.evaluate(self.preprocessing(X_test),y_test, batch_size=128)
        return score
