#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: Manuel Arnau Fernández, Marina Palomar González & Alba Fernández Coronado
"""

from abc import ABC, abstractmethod


class Visitor(ABC):   # abstract class that acts as interface, visitor pattern
    @abstractmethod
    def visit_leaf(self):
        pass

    @abstractmethod
    def visit_parent(self):
        pass


class FeatureImportance(Visitor):

    def __init__(self, dictionary_occurences_features, num_features):
        self.dictionary_occurences_features = dictionary_occurences_features
        # it if given a void dictionary
        self.num_features = num_features

        for i in range(num_features):
            dictionary_occurences_features[i] = 0
            # initializace each feature in the dictionary

    def visit_leaf(self):
        pass

    def visit_parent(self, Parent):

        self.dictionary_occurences_features[Parent.feature_index] += 1
        # increments the index given in the dictionary
        Parent.left_child.accept_visitor(self)
        Parent.right_child.accept_visitor(self)

    def print_occurences(self):
        print(self.dictionary_occurences_features)


class Printer(Visitor):

    def visit_leaf(self, Leaf):
        print('\t'*Leaf.depth + 'Leaf, label {}'.format(Leaf.label))

    def visit_parent(self, Parent):
        print('\t'*Parent.depth + 'parent, feature index {}, threshold {}'
              .format(Parent.feature_index, Parent.threshold))
        # '\t'*3 = '\t\t\t', with '\t'=tab
        Parent.left_child.accept_visitor(self)
        Parent.right_child.accept_visitor(self)