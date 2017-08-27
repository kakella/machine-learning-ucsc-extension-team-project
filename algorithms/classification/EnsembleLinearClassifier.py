import numpy as np
import random
import algorithms.classification.MeanSquareErrorMinimizerLinearClassifier as ln
import algorithms.util.misc as ut


class EnsembleLinearClassifier:

    def __init__(self, init_value=None):
        self.init_value = init_value or 1
        self.linearClassifier = None
        self.W = None

    @staticmethod
    def __generate_linear_classifiers(nd_data, n_classifiers):
        classifiers = np.random.uniform(-1, 1, (n_classifiers, nd_data.shape[1]))
        norms = np.linalg.norm(classifiers, axis=1)
        classifiers = ut.safe_divide(classifiers, norms)
        w0 = [-np.dot(classifiers[i], random.choice(nd_data).T) for i in range(n_classifiers)]
        return np.column_stack((w0, classifiers))

    @staticmethod
    def __generate_augmented_features(nd_data, init_value):
        return np.column_stack(([init_value]*nd_data.shape[0], nd_data))

    def __generate_meta_features(self, nd_data, n_classifiers):
        self.W = EnsembleLinearClassifier.__generate_linear_classifiers(nd_data, n_classifiers)
        Xa = EnsembleLinearClassifier.__generate_augmented_features(nd_data, self.init_value)
        return np.tanh(np.dot(Xa, self.W.T))

    def train(self, nd_data, T, n_classifiers):
        X = self.__generate_meta_features(nd_data, n_classifiers)
        self.linearClassifier = ln.MeanSquareErrorMinimizerLinearClassifier(self.init_value)
        self.linearClassifier.train(X, T)

    def classify(self, v_query):
        Xa = np.tanh(np.dot(EnsembleLinearClassifier.__generate_augmented_features(np.array([v_query]), self.init_value),
                            self.W.T))
        return self.linearClassifier.classify(Xa[0])
