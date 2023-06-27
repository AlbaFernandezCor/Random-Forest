#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: Manuel Arnau Fernández, Marina Palomar González & Alba Fernández Coronado
"""

from abc import ABC, abstractmethod
from visitor import FeatureImportance


class Node(ABC):   # interface for strategy pattern
    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def accept_visitor(self, Visitor):  # abstract method for Visitor pattern
        pass


class Leaf(Node):
    def __init__(self, label, depth):  # constructor of the Leaf
        self.label = label
        self.depth = depth

    def predict(self, x):
        return self.label

    def accept_visitor(self, Visitor):  # it receives an object of type Visitor
        if type(Visitor) == FeatureImportance:
            # leaf does not have features
            pass
        else:
            Visitor.visit_leaf(self)


class Parent(Node):
    def __init__(self, feature_index, threshold,
                 left_child, right_child, depth):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.depth = depth

    def predict(self, x):
        if (x[self.feature_index] < self.threshold):
            return self.left_child.predict(x)
        else:
            return self.right_child.predict(x)

    def accept_visitor(self, Visitor):  # it receives an object of type Visitor
        Visitor.visit_parent(self)