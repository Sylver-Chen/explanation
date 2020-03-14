import collections
import copy
from functools import partial

import numpy as np
import sklearn
import sklearn.preprocessing
from sklearn.utils import check_random_state

from explain import Explanation
from lime_base import LimeBase

class LimeTabularExplainer(object):
    def __init__(self,
                 training_data,
                 mode="classification",
                 #training_labels=None,
                 feature_names=None,
                 categorical_features=None,
                 categorical_names=None,
                 kernel_width=None,
                 kernel=None,
                 verbose=False,
                 class_names=None,
                 feature_selection='auto',
                 # discretize_continuous=True,
                 # discretizer='quartile',
                 sample_around_instance=False,
                 random_state=None,
                 # training_data_stats=None,
    ):
        """
        training_data: numpy 2d array
        random_state: an integer or numpy.RandomState that will be used to
            generate random numbers. If None, the random state will be
            initialized using the internal numpy seed.

        """
        self.random_state = check_random_state(random_state)
        self.mode = mode
        self.categorical_names = categorical_names or {}
        self.sample_around_instance = sample_around_instance

        if categorical_features is None:
            categorical_features = []
        if feature_names is None:
            feature_names = [str(i) for i in range(training_data.shape[1])]

        self.categorical_features = list(categorical_features)
        self.feature_names = list(feature_names)

        self.discretizer = None
        # self.training_data = training_data
        # self.feature_names = feature_names

        if kernel_width is None:
            kernel_width = np.sqrt(training_data.shape[1]) * .75
        kernel_width = float(kernel_width)

        if kernel is None:
            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        kernel_fn = partial(kernel, kernel_width=kernel_width)

        self.feature_selection = feature_selection
        self.base = LimeBase(kernel_fn, verbose, random_state=self.random_state)
        self.class_names = class_names

        self.scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        self.scaler.fit(training_data)
        self.feature_values = {}
        self.feature_frequencies = {}

        for feature in self.categorical_features:
            column = training_data[:, feature]
            feature_count = collections.Counter(column)
            values, frequencies = map(list, zip(*(sorted(feature_count.items()))))
            self.feature_values[feature] = values
            self.feature_frequencies[feature] = (np.array(frequencies) /
                                                 float(sum(frequencies)))
            self.scaler.mean_[feature] = 0
            self.scaler.scale_[feature] = 1

    @staticmethod
    def convert_and_round(values):
        return ['%.2f' % v for v in values]

    def explain_instance(
            self,
            data_row,
            predict_fn,
            labels=(1,),
            top_labels=None,
            num_features=10,
            num_samples=5000,
            distance_metric='euclidean',
            model_regressor=None
    ):
        """Generates explanations for a prediction.

        First, we generate neighborhood data by randomly perturbing features
        from the instance (see __data_inverse). We then learn locally weighted
        linear models on this neighborhood data.

        Args:
            data_row: 1d numpy array, corresponding to a row
            predict_fn: prediction function. For classifiers, this should be a
                function that takes a numpy array and outputs prediction
                probabilities.
            labels: iterable with labels to be explained.
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_samples: size of the neighborhood to learn the linear model
            distance_metric: the distance metric to use for weights.
            model_regressor: sklearn regressor to use in explanation. Defaults
                to Ridge regression in LimeBase. Must have model_regressor.coef_
                and 'sample_weight' as a parameter to model_regressor.fit()
        Returns:
            An Explanation object (see explanation.py) with the corresponding
            explanations.
        """
        data, inverse = self.__data_inverse(data_row, num_samples)

        scaled_data = (data - self.scaler.mean_) / self.scaler.scale_
        distances = sklearn.metrics.pairwise_distances(
                scaled_data,
                scaled_data[0].reshape(1, -1),
                metric=distance_metric
        ).ravel()
        yss = predict_fn(inverse)

        # for classification, the model needs to provide a list of tuples - classes
        # along with prediction probabilities
        if self.mode == "classification":
            if self.class_names is None:
                self.class_names = [str(x) for x in range(yss[0].shape[0])]
            else:
                self.class_names = list(self.class_names)
            if not np.allclose(yss.sum(axis=1), 1.0):
                warnings.warn("""
                Prediction probabilties do not sum to 1, and
                thus does not constitute a probability space.
                Check that you classifier outputs probabilities
                (Not log probabilities, or actual class predictions).
                """)
        else:
            raise NotImplementedError("regression not supported now")

        feature_names = copy.deepcopy(self.feature_names)
        if feature_names is None:
            feature_names = [str(x) for x in range(data_row.shape[0])]

        values = self.convert_and_round(data_row)
        feature_indexes = None

        # for i in self.categorical_features:
        #     name = int(data_row[i])
        #     if i in self.categorical_names:
        #         name = self.categorical_names[i][name]
        #     feature_names[i] = '%s=%s' % (feature_names[i], name)
        #     values[i] = 'True'
        # categorical_features = self.categorical_features

        ret_exp = Explanation(mode=self.mode,
                              class_names=self.class_names
        )
        ret_exp.scaled_data = scaled_data
        if self.mode == "classification":
            ret_exp.predict_proba = yss[0]
            if top_labels:
                # top label indices
                labels = np.argsort(yss[0])[-top_labels:]
                ret_exp.top_labels = list(labels)
                ret_exp.top_labels.reverse()

        # interpret
        for label in labels:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score,
             ret_exp.local_pred,
             ret_exp.right,
            ) = self.base.explain_instance_with_data(
                    scaled_data,
                    yss,
                    distances,
                    label,
                    num_features,
                    model_regressor=model_regressor,
                    feature_selection=self.feature_selection)

        return ret_exp


    def __data_inverse(self,
                       data_row,
                       num_samples):
        """Generates a neighborhood around a prediction.
        only numerical features for now!
        For numerical features, perturb them by sampling from a Normal(0,1) and
        doing the inverse operation of mean-centering and scaling, according to
        the means and stds in the training data.
        Args:
            data_row: 1d numpy array, corresponding to a row
            num_samples: size of the neighborhood to learn the linear model
        Returns:
            data: dense num_samples * [num_features] matrix. The first row is the
        original instance.
        """
        num_cols = data_row.shape[0]
        data = np.zeros((num_samples, num_cols))

        categorical_features = range(num_cols)

        instance_sample = data_row
        scale = self.scaler.scale_
        mean = self.scaler.mean_

        data = self.random_state.normal(
            0, 1, num_samples * num_cols).reshape(
                num_samples, num_cols
            )
        # sample around instance
        data = data * scale + instance_sample

        categorical_features = self.categorical_features
        first_row = data_row

        data[0] = data_row.copy()
        inverse = data.copy()

        for column in categorical_features:
            values = self.feature_values[column]
            freqs = self.feature_frequencies[column]
            inverse_column = self.random_state.choice(values, size=num_samples,
                                                      replace=True, p=freqs)
            binary_column = (inverse_column == first_row[column]).astype(int)
            binary_column[0] = 1
            inverse_column[0] = data[0, column]
            data[:, column] = binary_column
            inverse[:, column] = inverse_column

        inverse[0] = data_row

        return data, inverse
