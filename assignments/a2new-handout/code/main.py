#!/usr/bin/env python
import argparse
from cProfile import label
import os
import pickle
from pathlib import Path
from statistics import mode
from attr import validate
from matplotlib.axis import XAxis
import sys as sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sympy import beta
# from sklearn.tree import DecisionTreeClassifier

# make sure we're working in the directory this file lives in,
# for imports and for simplicity with relative paths
os.chdir(Path(__file__).parent.resolve())

# our code
from utils import load_dataset, plot_classifier, handle, run, main
from decision_stump import DecisionStumpInfoGain
from decision_tree import DecisionTree
from kmeans import Kmeans
from knn import KNN
from naive_bayes import NaiveBayes, NaiveBayesLaplace
from random_tree import RandomForest, RandomTree


@handle("1")
def q1():
    dataset = load_dataset("citiesSmall.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]
    """YOUR CODE HERE FOR Q1"""
    for k in [1,3,10]:
        model = KNN(k)
        model.fit(X, y)
        y_pred = model.predict(X)
        training_error =  np.mean(y_pred != y)
        test_error = np.mean(model.predict(X_test) != y_test)
        print("k=", k, ": training error=", training_error, ", test error=", test_error)
        if (k == 1):
            plot_classifier(model, X, y)
            plt.savefig(Path("..", "figs", "q1.pdf"))



@handle("2")
def q2():
    dataset = load_dataset("ccdebt.pkl")
    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]

    ks = list(range(1, 30, 4))

    """YOUR CODE HERE FOR Q2"""
    n = y.shape[0]

    # validation errors for different ks
    cv_accs = []
    for k in ks:
        validation_error = np.zeros(10)
        for i in range(10):
            mask = np.ones(n, dtype=bool)
            mask[int(0.1*i*n):int(0.1*(i+1)*n)] = False
            X_validate = X[~mask, :]
            y_validate = y[~mask]
            X_train = X[mask, :]
            y_train = y[mask]
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(X_train, y_train)
            validation_error[i] = np.mean(model.predict(X_validate) != y_validate)
        cv_accs.append(np.mean(validation_error))
    
    # test errors for different ks
    test_errors = []
    training_errors = []
    for k in ks:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X, y)
        test_error = np.mean(model.predict(X_test) != y_test)
        training_error = np.mean(model.predict(X) != y)
        test_errors.append(test_error)
        training_errors.append(training_error)
    print(cv_accs)
    print(test_errors)
    plt.plot(ks, cv_accs, label='cross validation' )
    plt.plot(ks, test_errors, label='test')
    plt.xlabel("k")
    plt.ylabel("error")
    plt.legend()
    plt.savefig(Path("..", "figs", "q2_1.pdf"))

    plt.plot(ks, training_errors, label='training')
    plt.legend()
    plt.savefig(Path("..", "figs", "q2_2.pdf"))
    



@handle("3.2")
def q3_2():
    dataset = load_dataset("newsgroups.pkl")

    X = dataset["X"].astype(bool)
    y = dataset["y"]
    X_valid = dataset["Xvalidate"]
    y_valid = dataset["yvalidate"]
    groupnames = dataset["groupnames"]
    wordlist = dataset["wordlist"]

    """YOUR CODE HERE FOR Q3.2"""
    print(groupnames)
    print(wordlist[72])
    print(wordlist[X[802]])
    print(groupnames[y[802]])



@handle("3.3")
def q3_3():
    dataset = load_dataset("newsgroups.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_valid = dataset["Xvalidate"]
    y_valid = dataset["yvalidate"]

    print(f"d = {X.shape[1]}")
    print(f"n = {X.shape[0]}")
    print(f"t = {X_valid.shape[0]}")
    print(f"Num classes = {len(np.unique(y))}")

    model = NaiveBayes(num_classes=4)
    model.fit(X, y)

    y_hat = model.predict(X)
    err_train = np.mean(y_hat != y)
    print(f"Naive Bayes training error: {err_train:.3f}")

    y_hat = model.predict(X_valid)
    err_valid = np.mean(y_hat != y_valid)
    print(f"Naive Bayes validation error: {err_valid:.3f}")


@handle("3.4")
def q3_4():
    dataset = load_dataset("newsgroups.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_valid = dataset["Xvalidate"]
    y_valid = dataset["yvalidate"]

    print(f"d = {X.shape[1]}")
    print(f"n = {X.shape[0]}")
    print(f"t = {X_valid.shape[0]}")
    print(f"Num classes = {len(np.unique(y))}")

    model = NaiveBayes(num_classes=4)
    model.fit(X, y)
    print("original:", model.p_xy[:,0])

    """YOUR CODE HERE FOR Q3.4"""
    model_laplace = NaiveBayesLaplace(num_classes=4, beta=10000)
    model_laplace.fit(X, y)
    print("laplace: ", model_laplace.p_xy[:, 0])

    print("diff: ", model_laplace.p_xy[:, 0] - model.p_xy[:,0])
    y_hat = model_laplace.predict(X_valid)
    err_valid = np.mean(y_hat != y_valid)
    print(f"Naive Bayes validation error: {err_valid:.3f}")


@handle("4")
def q4():
    dataset = load_dataset("vowel.pkl")
    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]
    print(f"n = {X.shape[0]}, d = {X.shape[1]}")

    def evaluate_model(model):
        model.fit(X, y)

        y_pred = model.predict(X)
        tr_error = np.mean(y_pred != y)

        y_pred = model.predict(X_test)
        te_error = np.mean(y_pred != y_test)
        print(f"    Training error: {tr_error:.3f}")
        print(f"    Testing error: {te_error:.3f}")

    print("Decision tree info gain")
    evaluate_model(DecisionTree(max_depth=np.inf, stump_class=DecisionStumpInfoGain))

    """YOUR CODE FOR Q4"""
    evaluate_model(RandomTree(max_depth=np.inf))
    evaluate_model(RandomForest(50, np.inf))



@handle("5")
def q5():
    X = load_dataset("clusterData.pkl")["X"]

    model = Kmeans(k=4)
    model.fit(X)
    y = model.predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="jet")

    fname = Path("..", "figs", "kmeans_basic_rerun.png")
    plt.savefig(fname)
    print(f"Figure saved as {fname}")


@handle("5.1")
def q5_1():
    X = load_dataset("clusterData.pkl")["X"]

    model = Kmeans(k=4)
    model.fit(X)
    y = model.predict(X)

    error = model.error(X, y, model.means)




@handle("5.2")
def q5_2():
    X = load_dataset("clusterData.pkl")["X"]

    min_error = sys.maxsize
    """YOUR CODE HERE FOR Q5.2"""
    for i in range(50):
        model = Kmeans(k=4)
        model.fit(X)
        y = model.predict(X)
        error = model.error(X, y, model.means)
        if error < min_error:
            y_min = y
            min_error = error
    
    print("min_error:", min_error)
    plt.scatter(X[:,0], X[:, 1], c=y_min, cmap="jet")
    fname = Path("..", "figs", "kmeans_50_run.png")
    plt.savefig(fname)

@handle("5.3")
def q5_3():
    X = load_dataset("clusterData.pkl")["X"]

    min_error = sys.maxsize
    min_errors = []

    for k in range(1, 11):
        for _ in range(50):
            model = Kmeans(k)
            model.fit(X)
            y = model.predict(X)
            error = model.error(X, y, model.means)
            if error < min_error:
                y_min = y
                min_error = error    
        min_errors.append(min_error)

    plt.plot([1,2,3,4,5,6,7,8,9,10], min_errors)
    plt.savefig(Path("..", "figs", "k-means-best-k.png"))

if __name__ == "__main__":
    main()

