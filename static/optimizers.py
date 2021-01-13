#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 12/28/20 11:51 下午
# @Author  : pcw
# @File    : optimizers.py
# @Description: <>
import numpy as np
from typing import NoReturn


class Optimizer:
    """模型参数优化器"""
    def update(self, *args, **kwargs):
        raise NotImplementedError


class SGD(Optimizer):
    """ stochastic gradient descent
    """
    def __init__(self, lr: float=1e-3):
        self.lr = lr

    def update(self, weight: np.ndarray, grad: np.ndarray) -> NoReturn:
        weight -= self.lr * grad

if __name__ == '__main__':
    sgd = SGD(0.1)
    a = np.ones((1,2))
    b = np.ones((1,2))
    sgd.update(a, b)
    print(a)
