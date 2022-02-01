"""
Implementation of k-nearest neighbours classifier
"""

import numpy as np

import utils
from utils import euclidean_dist_squared


class KNN:
    X = None
    y = None

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X  # just memorize the training data
        self.y = y

    def predict(self, X_hat):
        e_dist_squared = utils.euclidean_dist_squared(self.X, X_hat)
        sorted_indices = np.argsort(e_dist_squared, axis = 0)
        
        out = np.zeros(X_hat.shape[0])
        for i in range(sorted_indices.shape[1]):
            k_indices = sorted_indices[:self.k,i]
            values = np.fromiter((self.y[j] for j in k_indices), int)
            
            if np.sum(values) > self.k/2:
                out[i] = 1

        return out