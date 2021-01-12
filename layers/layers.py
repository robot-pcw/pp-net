#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 12/28/20 11:52 下午
# @Author  : pcw
# @File    : layers.py
# @Description: <>
import numpy as np
from core.tensor import Tensor
from typing import NoReturn
from core.activations import ActivationFunc

class Layer:
    """神经网络层
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
    def __init__(self, layer_name: str="param_layer") -> NoReturn:
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
    """非参数层：激活层
    """
    def __init__(self, activation_func: ActivationFunc, ):
        super().__init__(activation_func.name)
        self.func = activation_func

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.active = inputs
        return self.func.active_func(inputs)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        x -> h(x) -> y, h为激活函数, E为loss
        grad = (dE/dy)*(dy/dx) = grad_bp * h'
        """
        return self.func.grad_func(self.active) * grad


class Linear(ParamLayer):
    """线性层
    """
    def __init__(self, input_dim: int, output_dim: int):
        """
        Args:
            input_dim: 输入维度 n
            output_dim: 输出维度 m
        """
        super().__init__(layer_name="linear")
        self.weights =  super().weights_init((input_dim, output_dim))  # (n,m)
        self.bias = super().bias_init((output_dim, ))  # (1,m)
        self.weights_grad = Tensor(np.zeros_like(self.weights.data))  # (n,m)
        self.bias_grad = Tensor(np.zeros_like(self.bias_grad.data))  # (1,m)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        output = input @ w + b
        ---
        x1 -> | w1_1 w1_2 w1_3 |
        x2 -> | w2_1 w2_2 w2_3 |
        x3 -> | w3_1 w3_2 w3_3 |
        x4 -> | w4_1 w4_2 w4_3 |
                 +    +    +
        b ->  |  b1   b2   b3  |
                  v    v    v
        y     |  y1   y2   y3  |
        ---
        一个n维输入的m个仿射函数
        """
        self.inputs = inputs  # (1,n)
        return inputs @ self.weights + self.bias

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        grad = dE/dy = [dE/dy_1, ..., dE/dy_m]
        layer_input: x = [x1, ..., xn]
        layer_output: y = wx+b = x@w+b = [y1, ..., ym]
        线性层中，误差产生于与预期不符的输出值，当前层的输出依赖于层参数(w/b)以及层输入(上一层的输出x)
        为了减小误差，我们根据梯度的指导来调整参数
        其中当前层的参数调整由 dE/dw 和 dE/db 所确定
         1) dE/dw_ij = (dE/dy_j)*(dy_j/dw_ij) = grad_j * x_i
            --> dE/dw = x^T @ grad
         2) dE/db_j = (dE/dy_j)*(dy_j/db_j) = grad_j
            --> dE/db = grad
        上层的参数将根据dE/dx进行反向传递调整
         3) dE/dx_i = sum_j{(dE/dy_j)*(dy_j/dx_i)} = sum_j{grad_j * wij}
            --> dE/dx = (dE/dy)*(dy/dx) = grad @ w^T
        """
        self.weights_grad = self.inputs.T @ grad
        self.bias_grad = np.sum(grad, axis=0)
        return grad @ self.weights.T


if __name__ == '__main__':
    pass


