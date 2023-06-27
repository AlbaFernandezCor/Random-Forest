#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: Manuel Arnau Fernández, Marina Palomar González & Alba Fernández Coronado
"""

import numpy as np
from abc import ABC, abstractmethod


class Impurity_Measure(ABC):  # interface for strategy pattern
    @abstractmethod
    def compute(self, dataset):
        pass


class Gini(Impurity_Measure):
    def compute(self, dataset):
        impurity = 1
        list_of_classes = np.unique(dataset.y)  # [0,1,...,n]
        for i in range(len(list_of_classes)):
            elements = 0  # number of elements of each class
            for j in range(dataset.num_samples):
                if dataset.y[j] == list_of_classes[i]:
                    elements += 1
            impurity -= (elements/dataset.num_samples)**2  # impurity - (pc)**2
        return impurity


class Entropy(Impurity_Measure):
    def compute(self, dataset):
        entropy = 1
        list_of_classes = np.unique(dataset.y)  # [0,1,...,n]
        for i in range(len(list_of_classes)):
            elements = 0  # number of elements of each class
            for j in range(dataset.num_samples):
                if dataset.y[j] == list_of_classes[i]:
                    elements += 1
            entropy -= (elements/dataset.num_samples) *\
                np.log(elements/dataset.num_samples)  # entropy - pc*log(pc)
        return entropy


class MSE(Impurity_Measure):
    def compute(self, dataset):
        mean_samples_values = np.sum(dataset.y)/dataset.num_samples
        error = 0
        for y in dataset.y:
            error += (y-mean_samples_values)**2  # error + (y-ym)**2
        return error