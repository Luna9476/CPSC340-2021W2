from turtle import shape
from matplotlib.pyplot import axis
import numpy as np
from sympy import beta


class NaiveBayes:
    """
    Naive Bayes implementation.
    Assumes the feature are binary.
    Also assumes the labels go from 0,1,...k-1
    """

    p_y = None
    p_xy = None

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def fit(self, X, y):
        n, d = X.shape

        # Compute the number of class labels
        k = self.num_classes

        # Compute the probability of each class i.e p(y==c), aka "baseline -ness"
        counts = np.bincount(y)
        p_y = counts / n

        """YOUR CODE HERE FOR Q3.3"""

        # Compute the conditional probabilities i.e.
        # p(x_ij=1 | y_i==c) as p_xy[j, c]
        # p(x_ij=0 | y_i==c) as 1 - p_xy[j, c]
        p_xy = 0.5 * np.ones((d, k))
        xy = np.append(X, y[:,None], axis=1)

        # For each class, calculate the p(x_ij=1|y_i==b)
        for b in range(self.num_classes):
            # Find out where y_i==b
            indices = (xy[:, d]==b).nonzero()
            # summarize x_ij=1 for each word
            n_ij = np.sum(X[indices], axis=0)
            n_b = X[indices].shape[0]
            p_xy[:, b] = n_ij / n_b

        self.p_y = p_y
        self.p_xy = p_xy

    def predict(self, X):
        n, d = X.shape
        k = self.num_classes
        p_xy = self.p_xy
        p_y = self.p_y

        y_pred = np.zeros(n)
        for i in range(n):

            probs = p_y.copy()  # initialize with the p(y) terms
            for j in range(d):
                if X[i, j] != 0:
                    probs *= p_xy[j, :]
                else:
                    probs *= 1 - p_xy[j, :]

            y_pred[i] = np.argmax(probs)

        return y_pred


class NaiveBayesLaplace(NaiveBayes):
    def __init__(self, num_classes, beta=0):
        super().__init__(num_classes)
        self.beta = beta

    def fit(self, X, y):
        """YOUR CODE FOR Q3.4"""
        n,d = X.shape
        k = self.num_classes

        # append k [1,1,...1] and k [0,0...0] after original X
        X_append = np.ones((self.beta * k, d), dtype=int)
        X_append = np.append(X_append, np.zeros((self.beta * k, d)), axis=0)

        # append 2k [0,1,2,3] after original y
        y_append = np.repeat(np.array([i for i in range(k)]), self.beta *2).flatten()

        X = np.append(X, X_append, axis=0)
        y = np.append(y, y_append, axis=0)

        super(NaiveBayesLaplace, self).fit(X, y)
        
