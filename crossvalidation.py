#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: Manuel Arnau Fernández, Marina Palomar González & Alba Fernández Coronado
"""

import numpy as np


class CrossValidation():

    def __init__(self, num_folds):
        assert num_folds >= 2
        self.num_folds = num_folds

    def _group_folds(self, num_samples):
        size_fold = int(num_samples/self.num_folds)
        index_folds = []
        for i in range(self.num_folds):
            if (i != self.num_folds-1):
                index_folds.append([i*size_fold, (i+1)*size_fold])
            else:
                index_folds.append([i*size_fold, num_samples])
        return index_folds

    def split(self, X, y):
        # rotate train-validation partition to obtain better predictions
        num_samples = len(y)
        folds = self._group_folds(num_samples)
        i = 0
        while i < self.num_folds:
            X_folds_train = []
            y_folds_train = []
            for j in range(self.num_folds):
                if (j != i):
                    X_folds_train.append(X[folds[j][0]:folds[j][1]])
                    y_folds_train.append(y[folds[j][0]:folds[j][1]])
            X_train = np.vstack(X_folds_train)
            # Stack arrays in sequence vertically (row wise)
            X_test = X[folds[i][0]:folds[i][1]]
            y_train = np.hstack(y_folds_train)
            # Stack arrays in sequence horizontally (column wise)
            y_test = y[folds[i][0]:folds[i][1]]
            # not a return but a yield
            yield X_train, X_test, y_train, y_test
            i += 1