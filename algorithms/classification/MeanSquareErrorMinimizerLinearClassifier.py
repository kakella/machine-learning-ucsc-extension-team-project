import numpy as np
import algorithms.util.keslerization as ks


class MeanSquareErrorMinimizerLinearClassifier:

    def __init__(self, init_value=None):
        self.init_value = init_value or 1
        self.W = None
        self.class_labels = None

    @staticmethod
    def __generate_augmented_features(nd_data, init_value):
        return np.column_stack(([init_value]*nd_data.shape[0], nd_data))

    @staticmethod
    def __generate_pseudo_inverse(nd_data, init_value):
        Xa = MeanSquareErrorMinimizerLinearClassifier.__generate_augmented_features(nd_data, init_value)
        return np.linalg.pinv(Xa)

    @staticmethod
    def __generate_quadratic_terms(X):
        X_quad = X
        num_of_cols = X.shape[1]

        for m in range(num_of_cols):
            for n in range(m, num_of_cols):
                X_quad = np.column_stack((X_quad, X[:, m] * X[:, n]))

        return X_quad

    @staticmethod
    def __generate_cubic_terms(X):
        X_cubic = MeanSquareErrorMinimizerLinearClassifier.__generate_quadratic_terms(X)
        num_of_cols = X.shape[1]

        for m in range(num_of_cols):
            for n in range(m, num_of_cols):
                for o in range(n, num_of_cols):
                    X_cubic = np.column_stack((X_cubic, X[:, m] * X[:, n] * X[:, o]))

        return X_cubic

    def train(self, nd_data, T, non_linear_terms=None):
        if non_linear_terms == 'quadratic':
            X = MeanSquareErrorMinimizerLinearClassifier.__generate_quadratic_terms(nd_data)
        elif non_linear_terms == 'cubic':
            X = MeanSquareErrorMinimizerLinearClassifier.__generate_cubic_terms(nd_data)
        else:
            X = nd_data

        target, self.class_labels = ks.keslerize_column(T)
        self.W = np.dot(MeanSquareErrorMinimizerLinearClassifier.__generate_pseudo_inverse(X, self.init_value), target)

    def classify(self, v_query):
        T = np.dot(
            MeanSquareErrorMinimizerLinearClassifier.__generate_augmented_features(np.array([v_query]), self.init_value),
            self.W)
        return ks.de_keslerize_columns(T, self.class_labels)[0]
