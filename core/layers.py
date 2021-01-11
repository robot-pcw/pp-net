#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 12/28/20 11:52 下午
# @Author  : pcw
# @File    : layers.py
# @Description: <>
import numpy as np
from .tensor import Tensor
from typing import NoReturn
from core.activations import ActivationFunction

class Layer:
    """
    神经网络中的层，tensor ops
    """
    def __init__(self, name: str="layer"):
        self.layer_name = name

    def forward(self, inputs: Tensor) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad: Tensor) -> np.ndarray:
        raise NotImplementedError


class ParamLayer(Layer):
    """参数层
    """
    def __init__(self, layer_name: str="normal_layer") -> NoReturn:
        super().__init__(layer_name)
        self.weights: np.ndarray = None
        self.bias: np.ndarray = None

    def weights_init(self, dim_tuple: tuple) -> np.ndarray:
        return np.random.rand(dim_tuple)

    def bias_init(self, dim_tuple: tuple) -> np.ndarray:
        return np.random.rand(dim_tuple)

    def forward(self, inputs: Tensor) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad: Tensor) -> np.ndarray:
        raise NotImplementedError


class Activation(Layer):
    """非参数-激活层
    """
    def __init__(self, activation_func: ActivationFunction, ):
        super().__init__(activation_func.name)
        self.func = activation_func

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.active = inputs
        return self.func.active_func(inputs)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return self.func.grad_func(self.active) * grad


class Linear(ParamLayer):
    """
    线性层
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__(layer_name="linear")
        self.weights =  super().weights_init((input_dim, output_dim))
        self.bias = super().bias_init((output_dim, ))
        self.weights_grad = Tensor(np.zeros_like(self.weights.data))
        self.bias_grad = Tensor(np.zeros_like(self.bias_grad.data))

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        self.active = inputs @ self.weights + self.bias
        return self.active

    def backward(self, grad: Tensor) -> np.ndarray:
        pass



