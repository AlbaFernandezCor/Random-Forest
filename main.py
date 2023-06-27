#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 17:30:48 2021

@author: manu
"""

import Test as test
from crossvalidation import CrossValidation
import numpy as np
from impurity import Gini, Entropy


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


if __name__ == "__main__":
    print("Escull quin test vols executar:")
    print("1-test_regression")
    print("2-test_classifier_iris_gini")
    print("3-test_classifier_iris_entropy")
    print("4-test_classifier_iris_CV")
    print("5-test_mnist_extratrees")
    print("6-test_mnist_extratrees_multi")
    print("7-test_mnist_10")
    print("8-test_mnist_10_multi")
    opcio = int(input())

    if(opcio == 1):
        loaders = [sin, sum_sins]
        for loader in loaders:
            test.test_regression(loader)
    elif(opcio == 2):
        test.test_classifier_iris(Gini())
    elif(opcio == 3):
        test.test_classifier_iris(Entropy())
    elif(opcio == 4):
        kf = CrossValidation(5)
        test.test_classifier_iris_CV(kf)
    elif(opcio == 5):
        test.test_mnist(False, 1)
    elif(opcio == 6):
        test.test_mnist(True, 1)
    elif(opcio == 7):
        test.test_mnist(False, 10)
    elif(opcio == 8):
        test.test_mnist(True, 10)
    else:
        assert False, 'No test´{}'.format(opcio)
