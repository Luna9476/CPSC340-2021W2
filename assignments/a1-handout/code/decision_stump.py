import numpy as np
import utils


class DecisionStumpEquality:
    """
    This is a decision stump that branches on whether the value of X is
    "almost equal to" some threshold.

    This probably isn't a thing you want to actually do, it's just an example.
    """

    y_hat_yes = None
    y_hat_no = None
    j_best = None
    t_best = None

    def fit(self, X, y):
        n, d = X.shape

        # Get an array with the number of 0's, number of 1's, etc.
        count = np.bincount(y)

        # Get the index of the largest value in count.
        # Thus, y_mode is the mode (most popular value) of y
        y_mode = np.argmax(count)

        self.y_hat_yes = y_mode
        self.y_hat_no = None
        self.j_best = None
        self.t_best = None

        # If all the labels are the same, no need to split further
        if np.unique(y).size <= 1:
            return

        minError = np.sum(y != y_mode)

        # Loop over features looking for the best split
        for j in range(d):
            for i in range(n):
                # Choose value to equate to
                t = np.round(X[i, j])
                # Find most likely class for each split
                is_almost_equal = np.round(X[:, j]) == t
                y_yes_mode = utils.mode(y[is_almost_equal])
                y_no_mode = utils.mode(y[~is_almost_equal])  # ~ is "logical not"


                # Make predictions
                y_pred = y_yes_mode * np.ones(n)
                y_pred[np.round(X[:, j]) != t] = y_no_mode

                # Compute error
                errors = np.sum(y_pred != y)

                # Compare to minimum error so far
                if errors < minError:
                    # This is the lowest error, store this value
                    minError = errors
                    self.j_best = j
                    self.t_best = t
                    self.y_hat_yes = y_yes_mode
                    self.y_hat_no = y_no_mode

    def predict(self, X):
        n, d = X.shape
        X = np.round(X)

        if self.j_best is None:
            return self.y_hat_yes * np.ones(n)

        y_hat = np.zeros(n)

        for i in range(n):
            if X[i, self.j_best] == self.t_best:
                y_hat[i] = self.y_hat_yes
            else:
                y_hat[i] = self.y_hat_no

        return y_hat


class DecisionStumpErrorRate:
    y_hat_yes = None
    y_hat_no = None
    j_best = None
    t_best = None


    def fit(self, X, y):
        self.y_hat_yes = utils.mode(y)
        minError = np.sum(y != self.y_hat_yes)
        n, d = X.shape
        for j in range(d):
            thresholds = np.unique(X[:, j])
            for i in range(0, len(thresholds)):
                threshold = thresholds[i]

                # Find most likely class for each split
                y_yes_mode = utils.mode(y[X[:, j] > threshold])
                y_no_mode = utils.mode(y[X[:, j] <= threshold])

                # Make prediction
                y_pred = y_yes_mode * np.ones(n)
                y_pred[X[:, j] <= threshold] = y_no_mode

                # Compute errors
                errors = np.sum(y_pred != y)

                if errors < minError:
                    # This is the lowest error, store this value
                    
                    minError = errors
                    self.j_best = j
                    self.t_best = threshold
                    self.y_hat_yes = y_yes_mode
                    self.y_hat_no = y_no_mode

    def predict(self, X):
        n, d = X.shape
        if self.j_best is None:
            return self.y_hat_yes * np.ones(n)

        y_hat = np.zeros(n)

        for i in range(n):
            if (X[i, self.j_best] > self.t_best):
                y_hat[i] = self.y_hat_yes
            else:
                y_hat[i] = self.y_hat_no
        
        return y_hat


def entropy(p):
    """
    A helper function that computes the entropy of the
    discrete distribution p (stored in a 1D numpy array).
    The elements of p should add up to 1.
    This function ensures lim p-->0 of p log(p) = 0
    which is mathematically true, but numerically results in NaN
    because log(0) returns -Inf.
    """
    plogp = 0 * p  # initialize full of zeros
    plogp[p > 0] = p[p > 0] * np.log(p[p > 0])  # only do the computation when p>0
    return -np.sum(plogp)


class DecisionStumpInfoGain(DecisionStumpErrorRate):
    # This is not required, but one way to simplify the code is
    # to have this class inherit from DecisionStumpErrorRate.
    # Which methods (init, fit, predict) do you need to overwrite?
    y_hat_yes = None
    y_hat_no = None
    j_best = None
    t_best = None

    def fit(self, X, y):
        n, d = X.shape
        class_count = np.unique(y).size
        self.y_hat_yes = utils.mode(y)
        
        # If all ys are the same
        if (class_count == 1):
            return
        p = np.bincount(y, minlength = class_count) / n
        prev_entropy = entropy(p)
        maxInfo = 0

        for j in range(d):
            thresholds = np.unique(X[:, j])
            for i in range(0, len(thresholds)):
                threshold = thresholds[i]

                y_yes = y[X[:, j] > threshold]
                p_yes = np.bincount(y_yes, minlength=class_count) / len(y_yes)
                y_yes_mode = utils.mode(y_yes)


                y_no = y[X[:, j] <= threshold]
                p_no = np.bincount(y_no, minlength=class_count) / len(y_no)
                y_no_mode = utils.mode(y_no)

                # Make prediction
                y_pred = y_yes_mode * np.ones(n)
                y_pred[X[:, j] <= threshold] = y_no_mode

                new_entropy = len(y_yes) / n * entropy(p_yes) + len(y_no) / n * entropy(p_no)
                
                if prev_entropy - new_entropy > maxInfo:
                    maxInfo = prev_entropy - new_entropy
                    self.j_best = j
                    self.t_best = threshold
                    self.y_hat_yes = y_yes_mode
                    self.y_hat_no = y_no_mode
