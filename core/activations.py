#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 12/28/20 11:56 下午
# @Author  : pcw
# @File    : activations.py
# @Description: <>
from abc import ABCMeta, abstractmethod
import numpy as np

class ActivationFunction(metaclass=ABCMeta):
    name = "activation"
    @abstractmethod
    @classmethod
    def active_func(self, x):
        """激活函数
        """
        pass

    @abstractmethod
    @classmethod
    def grad_func(self, g):
        """激活函数的导数
        """
        pass


class Sigmoid(ActivationFunction):
    name = "sigmoid"
    def active_func(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def grad_func(self, g):
        ag = self.active_func(g)
        return ag*(1-ag)


class Tanh(ActivationFunction):
    name = "tanh"
    def active_func(self, x):
        return np.tanh(x)

    def grad_func(self, g):
        return 1.0 - np.power(np.tanh(g),2)


class ReLU(ActivationFunction):
    name = "relu"
    def active_func(self, x):
        return np.where(x<0, 0, x)

    def grad_func(self, g):
        return np.where(g<0, 0, 1)

if __name__ == '__main__':
    pass