#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 12/28/20 11:56 下午
# @Author  : pcw
# @File    : activations.py
# @Description: <>
import numpy as np
from typing import NamedTuple,Callable,Union


# ------激活函数枚举--------
active_param_types = Union[float,int]

class ActivationFunc(NamedTuple):
    name: str
    active_func: Callable[..., np.ndarray]
    grad_func: Callable[..., np.ndarray]


def sigmoid() -> ActivationFunc:
    return ActivationFunc("sigmoid", sigmoid_func, sigmoid_deriv)


def tanh() -> ActivationFunc:
    return ActivationFunc("tanh", tanh_func, tanh_deriv)


def relu() -> ActivationFunc:
    return ActivationFunc("relu", relu_func, relu_deriv)


def lrelu(alpha = .1) -> ActivationFunc:
    return ActivationFunc("leaky relu", lrelu_func(alpha), lrelu_deriv(alpha))


# ------激活函数--------
def sigmoid_func(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_deriv(g: np.ndarray) -> np.ndarray:
    act = sigmoid_func(g)
    return act * (1. - act)


def softmax_func(x: np.ndarray) -> np.ndarray:
    # 防止数值溢出
    z = x - np.max(x, axis=-1, keepdims=True)
    return _softmax_func(z)

def _softmax_func(x: np.ndarray) -> np.ndarray:
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

def softmax_deriv(g: np.ndarray) -> np.ndarray:
    pass


def tanh_func(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def tanh_deriv(g: np.ndarray) -> np.ndarray:
    return 1.0 - np.power(np.tanh(g), 2)


def relu_func(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, x, 0)


def relu_deriv(g: np.ndarray) -> np.ndarray:
    return np.where(g > 0, 1, 0)


def lrelu_func(alpha):
    def inner_func(x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, alpha*x)
    return inner_func

def lrelu_deriv(alpha):
    def inner_func(g: np.ndarray) -> np.ndarray:
        return np.where(g > 0, 1, alpha)
    return inner_func


if __name__ == '__main__':
    a = np.array([[1000, 500, 10, 1, 0.1], [100, 100, 10, 5, 1]])
    b = softmax_func(a)
    print(b)
    #print(np.max(a, axis=-1, keepdims=True))