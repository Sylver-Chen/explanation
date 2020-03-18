import math
import numpy as np
from sklearn.preprocessing import StandardScaler

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

def data_preprocessing(X):
    X_list = split_features(X)
    return X_list

def predict_proba(model, x):
    x = data_preprocessing(x)
    result = model.predict(x)#.flatten()
    return result

def get_embedding_model(model):
    """
    first layer to embedding layer.
    """
    # hard coded now
    layer_name = 'fully-connected-1'
    previous_layer_name = 'embedding'
    # first part model
    embedding_model = KerasModel(inputs=model.input,
                                          outputs=model.get_layer(previous_layer_name).output)
    return embedding_model

def get_remaining_model(model):
    """
    embedding layer to ouput layer.
    """
    # hard coded now
    layer_name = 'fully-connected-1'
    previous_layer_name = 'embedding'
    # first part model
    first_model = KerasModel(inputs=model.input,
                                          outputs=model.get_layer(previous_layer_name).output)

    # second part model
    layer_names = [layer.name for layer in model.layers]
    layer_idx = layer_names.index(layer_name)
    # get the input shape of the desired layer
    input_shape = model.layers[layer_idx].get_input_shape_at(0)
    new_input = Input(shape=(input_shape[1],))
    # create the new nodes for each layer in the path
    x = new_input
    for layer in model.layers[layer_idx:]:
        x = layer(x)
    remaining_model = KerasModel(new_input, x)

    return remaining_model

def get_encoder_and_decoder(model, training_data, num_hidden,
                            epochs=50, batch_size=128):
    # hard coded now
    layer_name = 'fully-connected-1'
    previous_layer_name = 'embedding'
    # encoder
    encoder = KerasModel(inputs=model.input, outputs=model.get_layer(previous_layer_name).output)
    # get the input for decoder
    embedding_vectors = encoder.predict(data_preprocessing(training_data))

    # decoder
    layer_names = [layer.name for layer in model.layers]
    layer_idx = layer_names.index(layer_name)
    # get the input shape of the desired layer
    input_shape = model.layers[layer_idx].get_input_shape_at(0)
    new_input = Input(shape=(input_shape[1],))

    output_model = Dense(num_hidden, kernel_initializer="uniform", activation='relu',
                         name='fully-connected')(new_input)
    output_model = Dense(len(model.input), activation='relu',
                         name='output')(output_model)
    decoder = KerasModel(inputs=new_input, outputs=output_model)
    # train decoder
    decoder.compile(loss='mean_squared_error', optimizer='adam',
                    metrics = ['accuracy']
    )
    decoder.fit(embedding_vectors, training_data,
                epochs=epochs, batch_size=batch_size
    )
    return encoder, decoder


class NN_with_EntityEmbedding(KerasModel):
    def __init__(self, X_train, y_train, # X_val, y_val,
                 categorical_features, categorical_names, class_names,
                 epochs=10,
                 batch_size=128,
                 mode='classification',
    ):
        super().__init__()
        print('init nn with embedding')
        self.categorical_features = categorical_features
        self.categorical_names = categorical_names
        self.class_names = class_names
        self.epochs = epochs
        self.batch_size = batch_size
        self.mode = mode
        self.__build_keras_model()
        self.fit(X_train, y_train)

    def __build_keras_model(self):
        input_model, output_embeddings = [], []
        for col in self.categorical_features:
            # determine number of different values of each column
            input_dim = len(self.categorical_names[col])
            # empirical [2017, Wang]
            output_dim = 6 * int(math.pow(input_dim, 1/4))
            input = Input(shape=(1,))
            output = Embedding(input_dim, output_dim, name='embedding_'+str(col))(input)
            output = Reshape(target_shape=(output_dim, ))(output)

            input_model.append(input)
            output_embeddings.append(output)

        output_model = Concatenate(name='embedding')(output_embeddings)
        output_model = Dense(1000, kernel_initializer="uniform", name='fully-connected-1')(output_model)
        output_model = Activation('relu')(output_model)
        output_model = Dense(500, kernel_initializer="uniform", name='fully-connected-2')(output_model)
        # output_model = Dense(10, kernel_initializer="uniform", name='fully-connected')(output_model)
        output_model = Activation('relu')(output_model)
        if self.mode == 'classification':
            output_model = Dense(len(self.class_names), name='output')(output_model)
            output_model = Activation('softmax')(output_model)
        else:
            output_model = Dense(1, name='output')(output_model)
            output_model = Activation('sigmoid')(output_model)

        self.model = KerasModel(inputs=input_model, outputs=output_model)

        if self.mode == 'classification':
            self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['sparse_categorical_accuracy'])
        else:
            self.model.compile(loss='mean_absolute_error', optimizer='adam')

    def fit(self, X_train, y_train):
        self.model.fit(data_preprocessing(X_train),
                       y_train,
                       epochs=self.epochs, batch_size=self.batch_size,)

    def predict_proba(self, x):
        x = data_preprocessing(x)
        result = self.model.predict(x)#.flatten()
        return result

    def evaluate(self, X_test, y_test):
        # return the loss value & metric values for the model in test mode.
        score = self.model.evaluate(data_preprocessing(X_test),y_test, batch_size=None)
        return score


class NN(KerasModel):
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


    def __build_keras_model(self):
        self.model = Sequential()
        input_dim = len(self.categorical_features)
        # # alternative to embedding layer
        alternative_dim = 0
        for col in self.categorical_features:
            size = len(self.categorical_names[col])
            alternative_dim += 6 * int(math.pow(size, 1/4))
        self.model.add(Dense(alternative_dim, kernel_initializer="uniform",
                             input_dim=input_dim))
        self.model.add(Activation('relu'))
        # same as nn with embedding
        self.model.add(Dense(1000, kernel_initializer="uniform",
                             input_dim=input_dim)
        )

        self.model.add(Dense(1000, kernel_initializer="uniform",))
        self.model.add(Activation('relu'))
        self.model.add(Dense(500, kernel_initializer="uniform"))
        # self.model.add(Dense(10, kernel_initializer="uniform"))
        self.model.add(Activation('relu'))

        if self.mode == 'classification':
            self.model.add(Dense(len(self.class_names), kernel_initializer="uniform"))
            self.model.add(Activation('softmax'))
        else:
            self.model.add(Dense(1))
            self.model.add(Activation('sigmoid'))

        if self.mode == 'classification':
            self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['sparse_categorical_accuracy'])
        else:
            self.model.compile(loss='mean_absolute_error', optimizer='adam')

    def fit(self, X_train, y_train):
        self.model.fit(X_train,
                       y_train,
                       epochs=self.epochs, batch_size=self.batch_size,)

    def predict_proba(self, x):
        result = self.model.predict(x)#.flatten()
        return result

    def evaluate(self, X_test, y_test):
        # return the loss value & metric values for the model in test mode.
        score = self.model.evaluate(X_test, y_test, batch_size=None)
        return score
