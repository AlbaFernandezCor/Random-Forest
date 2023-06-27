#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: Manuel Arnau Fernández, Marina Palomar González & Alba Fernández Coronado 
"""

import numpy as np
from tree import Parent, Leaf
from dataset import DataSet
from visitor import FeatureImportance, Printer
import time
from multiprocessing import Pool
from abc import ABC, abstractmethod


class RandomForest(ABC):
    def __init__(self, num_trees, min_size, max_depth, ratio_samples,
                 num_random_features, criterion,
                 optimization, multiprocessing):
        self.num_trees = num_trees
        self.min_size = min_size
        self.max_depth = max_depth
        self.ratio_samples = ratio_samples
        self.num_random_features = num_random_features
        self.criterion = criterion
        self.multiprocessing = multiprocessing
        self.optimization = optimization
        self.decision_trees = []

    def predict(self, X):  # pass some data and predict what class is
        ypred = []
        for x in X:
            predictions = [tree.predict(x) for tree in self.decision_trees]
            # list with the prediction of each tree made
            ypred.append(self._combine_predictions(predictions))
        return np.array(ypred)

    def fit(self, X, y):  # with the training data (X,y), build a dataset
        dataset = DataSet(X, y)
        self.num_features = dataset.num_features
        if(self.multiprocessing):
            self._make_decision_trees_multiprocessing(dataset)
        else:
            self._make_decision_trees(dataset)

    def _make_decision_trees(self, dataset):
        # replace the set with a subset
        for i in range(self.num_trees):
            subset = dataset.random_sampling(self.ratio_samples)
            # incial dataset row subset
            tree = self._make_node(subset, 1)  # creates roots
            self.decision_trees.append(tree)

    def _make_node(self, dataset, depth):
        if depth == self.max_depth\
            or dataset.num_samples <= self.min_size\
                or dataset.all_same_label():
            # last condition is true if all samples belong to the same class
            node = self._make_leaf(dataset, depth)
            # if the conditions pass, a leaf is created (new class)
        else:
            node = self._make_parent_or_leaf(dataset, depth)
            # however, the depth is increased
        return node

    def _make_parent_or_leaf(self, dataset, depth):
        # we make more diverse abbreviates with a random subset
        idx_features = np.random.choice(range(dataset.num_features),
                                        self.num_random_features,
                                        replace=False)
        index_samples = np.random.choice(range(dataset.num_samples),
                                         min(self.optimization,
                                             self.min_size), replace=False)
        # we are looking for a better partner (feature, threshold)
        assert len(index_samples) >= 1 and \
            len(index_samples) <= dataset.num_samples
        best_feature_index, best_threshold, min_cost, best_split = \
            np.Inf, np.Inf, np.Inf, None
        for idx2 in index_samples:
            for idx in idx_features:
                val = dataset.X[idx2, idx]
                left_dataset, right_dataset = dataset.split_dataset(idx, val)
                cost = self._CART_cost(left_dataset, right_dataset)
                if cost < min_cost:
                    best_feature_index, best_threshold, min_cost, best_split \
                        = idx, val, cost, [left_dataset, right_dataset]
        left_dataset, right_dataset = best_split
        assert left_dataset.num_samples > 0 or right_dataset.num_samples > 0
        if left_dataset.num_samples == 0 or right_dataset.num_samples == 0:
            return self._make_leaf(dataset, depth)
        else:
            left_child = self._make_node(left_dataset, depth + 1)
            right_child = self._make_node(right_dataset, depth + 1)
            node = Parent(best_feature_index, best_threshold,
                          left_child, right_child, depth)
            # as we are doing the children of a father, depht+1
            return node

    def _CART_cost(self, left_dataset, right_dataset):
        # calculates the impurity of the datasets
        impurity_left = self.criterion.compute(left_dataset)
        impurity_right = self.criterion.compute(right_dataset)
        total_samples = right_dataset.num_samples+left_dataset.num_samples
        # sum of samples that exist
        cost = (left_dataset.num_samples/total_samples)*impurity_left + \
            (right_dataset.num_samples/total_samples)*impurity_right
        return cost

    @abstractmethod
    def _make_leaf(self, dataset, depth):
        pass

    @abstractmethod
    def _combine_predictions(self, predictions):
        pass

    def feature_importance(self):
        occurences = {}
        features = FeatureImportance(occurences, self.num_features)
        assert len(occurences) > 0
        for tree in self.decision_trees:  # tree = rot node
            tree.accept_visitor(features)
        features.print_occurences()

    def print_trees(self):
        printer = Printer()
        count = 1
        for tree in self.decision_trees:
            print("decision tree", count)
            count += 1
            tree.accept_visitor(printer)  # tree = root node

    def _target(self, dataset, nproc):
        print('process {} starts'.format(nproc))
        subset = dataset.random_sampling(self.ratio_samples)
        tree = self._make_node(subset, 1)
        print('process {} ends'.format(nproc))
        return tree

    def _make_decision_trees_multiprocessing(self, dataset):
        t1 = time.time()
        with Pool() as pool:
            self.decision_trees = pool.starmap(self._target,
                                               [(dataset, nproc) for nproc
                                                in range(self.num_trees)])
            # if there is only one argument in _target we use pool. map
        t2 = time.time()
        print('{} seconds per tree'.format((t2 - t1)/self.num_trees))


class RandomForestClassifier(RandomForest):
    def _combine_predictions(self, predictions):
        assert len(predictions) > 0
        return max(set(predictions), key=predictions.count)

    def _make_leaf(self, dataset, depth):
        return Leaf(dataset.most_frequent_label(), depth)
        # returns the most frequent label


class RandomForestRegressor(RandomForest):
    def _combine_predictions(self, predictions):
        assert len(predictions) > 0
        return np.mean(predictions)

    def _make_leaf(self, dataset, depth):
        return Leaf(dataset.y.mean(), depth)