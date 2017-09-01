#!/usr/bin/env python3

import numpy as np
import algorithms.util.misc as ut
import matplotlib.pyplot as plt


class KMeansClusteringClassifier:

    def __init__(self, X, C=2, stop_threshold=None, n_iterations=None):
        self.stop_threshold = stop_threshold or 0.1
        self.n_iterations = n_iterations or 10000
        self.classifier = None
        self.X = X                      # N x d features
        self.N = X.shape[0]             # data samples
        self.d = X.shape[1]             # feature vectors
        self.C = C                      # guassian mixture components

        self.a = np.zeros(self.C)                       # component weight
        self.u = np.zeros((self.C,self.d))              # mean
        self.S = np.zeros((self.C,self.d,self.d))       # covariance
        self.P = np.zeros((self.N,self.C))              # posterior probabilities
        
        for k in range(self.C):
            self.a[k] = 1/self.C
            self.u[k] = np.random.rand(self.d)
            for i in range(self.d):
                self.u[k][i] *= (max(X[:,i])-min(X[:,i])) - abs(min(X[:,i]))
            self.S[k] = np.cov(X, rowvar=False)

    @staticmethod
    def __init_means(nd_data, k):
        t_shape = nd_data.shape
        v_min = np.array([min(nd_data[:, i]) for i in range(t_shape[1])])
        v_max = np.array([max(nd_data[:, i]) for i in range(t_shape[1])])
        return np.random.uniform(v_min, v_max, (k, t_shape[1]))

    @staticmethod
    def __per_mean_iterate(nd_data, v_mean):
        return [sum((feature - v_mean) ** 2) for feature in nd_data]

    @staticmethod
    def __calculate_errors(nd_data, nd_means):
        return [KMeansClusteringClassifier.__per_mean_iterate(nd_data, v_mean) for v_mean in nd_means]

    @staticmethod
    def __find_closest_mean(nd_errors):
        return np.argmin(nd_errors, axis=0)

    @staticmethod
    def __recalculate_means(nd_data, nd_closest_means, k):
        t_shape = nd_data.shape
        nd_means = np.zeros((k, t_shape[1]))

        v_mean_feature_counts = np.zeros(k)
        for i, index in enumerate(nd_closest_means):
            nd_means[index] += nd_data[i]
            v_mean_feature_counts[index] += 1

        nd_means = ut.safe_divide(nd_means, v_mean_feature_counts)
        return nd_means

    @staticmethod
    def __calculate_means_delta(old_means, new_means):
        delta = 0
        diff = np.square(ut.safe_divide((new_means - old_means), old_means))
        for mean in diff:
            delta += sum(mean)
        return delta

    def train(self, k, plot=False):
        nd_data = self.X
        nd_means = KMeansClusteringClassifier.__init_means(nd_data, k)
        recalculated_means = None

        means_tracker = nd_means
        iteration_tracker = 0
        red = 0x00
        green = 0xff
        blue = 0x00

        for i in range(1, self.n_iterations):
            iteration_tracker += 1
            nd_errors = KMeansClusteringClassifier.__calculate_errors(nd_data, nd_means)
            nd_closest_means = KMeansClusteringClassifier.__find_closest_mean(nd_errors)
            recalculated_means = KMeansClusteringClassifier.__recalculate_means(nd_data, nd_closest_means, k)
            means_delta = float(KMeansClusteringClassifier.__calculate_means_delta(nd_means, recalculated_means))

            if plot:
                print( means_delta )
                plt.plot(self.u[:,0], self.u[:,1], linestyle='--',color="#%02x%02x%02x"%(red,green,blue))
                red += 0x5
                red %= 0xff
                green -= 0x5
                if green < 0:
                    green = 0xff
                plt.pause(0.1)

            if means_delta <= self.stop_threshold:
                plt.plot(self.u[:,0], self.u[:,1], linestyle='-', linewidth=4, color="#%02x%02x%02x"%(red,green,blue))
                break
            else:
                nd_means = recalculated_means
                self.u = nd_means
                means_tracker = np.concatenate((means_tracker, nd_means))

        self.classifier = recalculated_means
        return means_tracker, iteration_tracker

    def classify(self, v_query):
        nd_errors = KMeansClusteringClassifier.__calculate_errors(np.array([v_query]), self.classifier)
        return KMeansClusteringClassifier.__find_closest_mean(nd_errors)[0]
