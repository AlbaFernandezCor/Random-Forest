#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: Manuel Arnau Fernández, Marina Palomar González & Alba Fernández Coronado
"""

import numpy as np


class DataSet():

    def __init__(self, X, y):
        try:
            self.num_samples, self.num_features = X.shape
            self.X = X
            self.y = y
        except:
            self.num_samples = len(y)
            self.X = X
            self.y = y

    def random_sampling(self, ratio_sampling):
        assert ratio_sampling <= 1 and ratio_sampling > 0
        # select a random index_list with replacement
        index_list = np.random.choice(self.num_samples,
                                      int(self.num_samples*ratio_sampling),
                                      replace=True)
        return DataSet(self.X[index_list], self.y[index_list])

    def all_same_label(self):
        if len(np.unique(self.y)) == 1:
            return True
            # it means they are all of the same class
        else:
            return False
            # if False, they are from different classes

    def split_dataset(self, index, threshold):
        idxleft = self.X[:, index] < threshold
        idxright = self.X[:, index] >= threshold

        return DataSet(self.X[idxleft],
                       self.y[idxleft]),\
            DataSet(self.X[idxright],
                    self.y[idxright])

    def most_frequent_label(self):
        return(np.bincount(self.y).argmax())
        # we return the label that appears the most