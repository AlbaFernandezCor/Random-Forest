# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 19:45:11 2021

@author: Pc
"""

from randomforest import RandomForestClassifier, RandomForestRegressor
import numpy as np
import sklearn.datasets
from impurity import Gini, MSE
from crossvalidation import CrossValidation
import matplotlib.pyplot as plt
import mnist as mnist


def test_regression(loader):
    X, y = loader()
    seed = 1
    np.random.seed(seed)
    num_samples = X.shape[0]
    max_depth = 10
    min_size_split = 5
    ratio_samples = 0.7
    num_trees = 10
    num_random_features = int(np.sqrt(X.shape[1]))
    ratio_train = 0.75
    idx = np.random.permutation(range(num_samples))
    num_samples_train = int(num_samples*ratio_train)
    idx_train = idx[:num_samples_train]
    idx_test = idx[num_samples_train:]
    idx_train = np.sort(idx_train)
    idx_test = np.sort(idx_test)
    X_train, y_train = X[idx_train], y[idx_train]
    X_test, y_test = X[idx_test], y[idx_test]
    print('Random forest regressor')
    X_train, y_train = X[idx_train], y[idx_train]
    X_test, y_test = X[idx_test], y[idx_test]
    rf = RandomForestRegressor(num_trees, min_size_split, max_depth,
                               ratio_samples, num_random_features, MSE(),
                               len(y), False)  # len(y) = optimization,
                                                # False = multiprocessing
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    # Root Mean Square Error
    rmse = np.sqrt(np.sum((y_pred - y_test) ** 2) / float(len(y_test)))
    # Mean Absolute Error
    mae = np.mean(np.abs(y_pred - y_test))
    print('rmse {}, mae {}'.format(np.round(rmse, decimals=3),
                                   np.round(mae, decimals=3)))
    # Plots all the regression graphics
    plt.figure()
    plt.plot(X_train, y_train, '.-')
    plt.title('train')
    plt.figure()
    plt.plot(X_test, y_test, 'g.-', label='test')
    plt.plot(X_test, y_pred, 'y.-', label='prediction')
    plt.legend()


def sum_sins():
    rng = np.random.RandomState(1)
    X = np.linspace(0, 6, 300)[:, np.newaxis]
    y = np.sin(X).ravel() + np.sin(6 * X).ravel() +\
        rng.normal(0, 0.1, X.shape[0])
    return X, y  # dataset to implement regression


def sin():
    rng = np.random.RandomState(1)
    X = np.sort(5 * rng.rand(200, 1), axis=0)
    y = np.sin(X).ravel()
    return X, y  # dataset to implement regression


def test_classifier_iris(criterion):
    max_depth = 10
    ratio_samples = 0.7
    num_trees = 10
    min_size_split = 5
    ratio_train = 0.7
    ratio_test = 1-ratio_train
    iris = sklearn.datasets.load_iris()
    X, y = iris.data, iris.target
    num_samples, num_features = X.shape
    # 150 rows = num_samples, 4 columns = num_features
    num_random_features = int(np.sqrt(num_features))
    idx = np.random.permutation(range(num_samples))
    num_samples_train = int(num_samples*ratio_train)
    num_samples_test = int(num_samples*ratio_test)
    idx_train = idx[:num_samples_train]
    idx_test = idx[num_samples_train:num_samples_train + num_samples_test]
    X_train, y_train = X[idx_train], y[idx_train]
    X_test, y_test = X[idx_test], y[idx_test]
    X, y = X[idx], y[idx]
    rf = RandomForestClassifier(num_trees, min_size_split, max_depth,
                                ratio_samples, num_random_features, criterion,
                                len(y), False)  # len(y) = optimization
                                                # False = multiprocessing
    rf.fit(X_train, y_train)
    ypred = rf.predict(X_test)
    num_samples_test = len(y_test)
    num_correct_predictions = np.sum(ypred == y_test)
    accuracy = num_correct_predictions/float(num_samples_test)
    print('accuracy {} %'.format(100*np.round(accuracy, decimals=2)))
    print("Feature Importance: \n")
    rf.feature_importance()
    print("\n")
    rf.print_trees()


def test_classifier_iris_CV(kf):
    max_depth = 10
    ratio_samples = 0.7
    num_trees = 10
    min_size_split = 5
    iris = sklearn.datasets.load_iris()
    X, y = iris.data, iris.target
    num_samples, num_features = X.shape
    # 150 rows = num_samples, 4 columns = num_features
    num_random_features = int(np.sqrt(num_features))
    idx = np.random.permutation(range(num_samples))
    X, y = X[idx], y[idx]
    kf = CrossValidation(5)  # Number of folds used to split the dataset
    accuracy = []
    for X_train, X_test, y_train, y_test in kf.split(X, y):
        rf = RandomForestClassifier(num_trees, min_size_split, max_depth,
                                    ratio_samples, num_random_features, Gini(),
                                    len(y), False)  # len(y) = optimization
                                                    # False = multiprocessing
        rf.fit(X_train, y_train)
        ypred = rf.predict(X_test)
        num_samples_test = len(y_test)
        num_correct_predictions = np.sum(ypred == y_test)
        accuracy.append(float(num_correct_predictions/float(num_samples_test)))
    accuracy = sum(accuracy)/len(accuracy)
    print('accuracy {} %'.format(100*np.round(accuracy, decimals=2)))
    print("Feature Importance: \n")
    rf.feature_importance()
    print("\n")
    rf.print_trees()


# We decide wether we use multiprocessing or not and the optimization we use
def test_mnist(multiprocessing, optimization):
    X_train, y_train, X_test, y_test = mnist.load()
    max_depth = 20
    num_trees = 1
    ratio_samples = 0.25
    min_size_split = 20
    num_samples, num_features = X_train.shape
    num_random_features = int(np.sqrt(num_features))
    rf = RandomForestClassifier(num_trees, min_size_split, max_depth,
                                ratio_samples, num_random_features, Gini(),
                                optimization, multiprocessing)
    rf.fit(X_train, y_train)
    ypred = rf.predict(X_test)
    num_samples_test = len(y_test)
    num_correct_predictions = np.sum(ypred == y_test)
    accuracy = num_correct_predictions/float(num_samples_test)
    print('accuracy {} %'.format(100*np.round(accuracy, decimals=2)))
