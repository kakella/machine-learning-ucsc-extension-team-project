#! /usr/bin/env python3


# Imports from example
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy.linalg as linalg
import math


class ExpectationMaximizationClassifier:
    """Expectation-Maximization algorithm."""

    def __init__(self, X, C=2):
        """Initialize variables"""

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

    def _E(self):
        """E step: find posterior probabilities"""

        for i in range(self.N):
            row_sum = 0
            for k in range(self.C):
                diff = self.X[i,:] - self.u[k]
                exp = np.dot(-0.5*diff, np.linalg.inv(self.S[k]))
                exp = np.dot(exp, diff.transpose())
                self.P[i,k] = self.a[k] * (math.e**exp)
                self.P[i,k] /= ( (2*math.pi)**(self.d/2) * math.sqrt(np.linalg.det(self.S[k])) )
                row_sum += self.P[i,k]
            self.P[i,:] /= row_sum

    def _M(self):
        """M step: find model parameters"""

        u_new = np.zeros((self.C,self.d))
        for k in range(self.C):
            p_sum = 0
            S_new = np.zeros((self.d,self.d))
            for i in range(self.N):
                p_sum += self.P[i,k]
                u_new[k] += self.P[i,k] * self.X[i,:]
                diff = self.X[i,:] - self.u[k]
                diff = diff.reshape((1,self.d))
                S_new += self.P[i,k] * np.dot(diff.transpose(), diff)

            self.a[k] = p_sum / self.N
            u_new[k] /= p_sum
            S_new /= p_sum
            self.S[k] = S_new
        error = np.max(abs(self.u - u_new))
        self.u = u_new

        return (error)

    def Run(self, iterations=10000, tolerance=10**-3, print_progress=False, plot=False):
        """Runs the algorithm."""

        u_orig = self.u
        red = 0x00
        green = 0xff
        blue = 0x00

        error = 10000
        for i in range(iterations):
            print("iterations: %d error: %f" % (i, error))
            for k in range(self.C):
                if print_progress:
                    print("  k", k)
                    print("  a", self.a[k], "u", self.u[k])

            self._E()
            error = self._M()

            if error < tolerance:
                print("EM end in success after %d iterations" % i)
                break

            if plot:
                #plt.scatter(self.u[:,0], self.u[:,1], color="#%02x%02x%02x"%(red,green,blue))
                plt.scatter(self.u[:,0], self.u[:,1], color="#%02x%02x%02x"%(red,green,blue))
                red += 0x5
                red %= 0xff
                green -= 0x5
                if green < 0:
                    green = 0xff
                plt.pause(0.1)

        if plot and False:
            plt.scatter(u_orig[:,0], u_orig[:,1], color='y')        # Initial
            plt.scatter(self.u[:,0], self.u[:,1], color='c')        # Final


