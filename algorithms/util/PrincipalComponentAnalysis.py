import numpy as np
import numpy.linalg as nl


# TODO: Add checks to ensure PCA intermediate steps and output are correct


class PrincipalComponentAnalysis:

    def __init__(self):
        None

    @staticmethod
    def __find_optimal_principals(eigenvalues, threshold):
        cumulative_sum = np.cumsum(eigenvalues)
        sum = cumulative_sum[-1]
        return next(ev[0] for ev in enumerate(cumulative_sum) if ev[1]/sum > threshold) + 1

    @staticmethod
    def pca(X, threshold=0.9):
        v_mean = np.mean(X, axis=0)
        Z = X - v_mean
        C = np.cov(Z, rowvar=False)

        eigenvalues, V = nl.eigh(C)
        eigenvalues = np.flipud(eigenvalues)
        V = np.flipud(V.T)

        num_of_pcs = PrincipalComponentAnalysis.__find_optimal_principals(eigenvalues, threshold)
        Vpc = V[:num_of_pcs]
        P = np.dot(Z, Vpc.T)
        return P, num_of_pcs
