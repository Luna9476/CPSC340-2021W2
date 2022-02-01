import imp
from itertools import count
from random import random
from scipy import stats
from matplotlib.pyplot import axis
from random_stump import RandomStumpInfoGain
from decision_tree import DecisionTree
import numpy as np

import utils


class RandomTree(DecisionTree):
    def __init__(self, max_depth):
        DecisionTree.__init__(
            self, max_depth=max_depth, stump_class=RandomStumpInfoGain
        )

    def fit(self, X, y):
        n = X.shape[0]
        boostrap_inds = np.random.choice(n, n, replace=True)
        bootstrap_X = X[boostrap_inds]
        bootstrap_y = y[boostrap_inds]

        DecisionTree.fit(self, bootstrap_X, bootstrap_y)


class RandomForest:
    """
    YOUR CODE HERE FOR Q4
    Hint: start with the constructor __init__(), which takes the hyperparameters.
    Hint: you can instantiate objects inside fit().
    Make sure predict() is able to handle multiple examples.
    """
    random_trees = []

    def __init__(self, num_trees, max_depth):
        self.num_trees = num_trees
        self.max_depth = max_depth

    
    def fit(self, X, y):
        for i in range(self.num_trees):
            random_tree = RandomTree(max_depth=self.max_depth)
            random_tree.fit(X, y)
            self.random_trees.append(random_tree)
    
    def predict(self, X):
        res = []
        mode = np.zeros(X.shape[0])
        
        for i in range(self.num_trees):
            y_pred = self.random_trees[i].predict(X)
            res.append(y_pred)
        
        res = np.array(res)
        for i in range(X.shape[0]):
            mode[i] = utils.mode(res[i, :])

        return mode

        


