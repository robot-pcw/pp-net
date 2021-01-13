#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 1/11/21 2:46 下午
# @Author  : pcw
# @File    : net.py
# @Description: <>
from typing import Callable, NoReturn, Optional, List
from static.layers import Layer, ParamLayer
from static.optimizers import Optimizer
from static.loss import Loss
import numpy as np
import time

class Net:
    """Neural Network"""
    def compile(self, optimizer: Optimizer, loss: Loss, metrics: Optional[Callable]=None):
        self.optimizer = optimizer
        self.loss_func = loss
        self.metrics = metrics if metrics else []

    def fit(self, *args, **kwargs) -> NoReturn:
        raise NotImplementedError

    def evaluate(self, *args, **kwargs) -> NoReturn:
        raise NotImplementedError

    def predict(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError


class SequentialNet(Net):
    """静态图"""
    def __init__(self):
        self.layers: List[Layer] = []

    def show(self):
        print("---> Sequential Neural Network")
        for lay in self.layers:
            print(lay)
        print("------------------------------")

    def add(self, layer: Layer):
        self.layers.append(layer)

    def fit(self, train_data: np.ndarray, train_label: np.ndarray, epochs: int=5):
        print("input size: {}, target size:{}".format(train_data.shape, train_label.shape))
        if len(train_label.shape)==1:
            train_label = np.expand_dims(train_label, axis=1)
        # todo: batch step
        for epoch in range(epochs):
            # forward propagation
            train_pred = self._forward_prop(train_data)

            # backward propagation
            batch_mean_loss = self.loss_func.batch_loss(train_pred, train_label)
            loss_grad = self.loss_func.loss_grad(train_pred, train_label)
            self._backward_prop(loss_grad)
            print("epoch:{}, train_loss:{} ".format(epoch, batch_mean_loss))


    def _forward_prop(self, x: np.ndarray) -> np.ndarray:
        last_input = x
        for layer in self.layers:
            last_input = layer.forward(last_input)
        return last_input

    def _backward_prop(self, loss_grad: np.ndarray) -> NoReturn:
        last_grad = loss_grad
        for layer in reversed(self.layers):
            #print("{} \n  bp grad: {}".format(layer, last_grad))
            last_grad = layer.backward(last_grad)
            if isinstance(layer, ParamLayer):
                for tensor in layer.params_to_update:
                    self.optimizer.update(tensor.data, tensor.grad)

    def evaluate(self, val_data: np.ndarray, val_label: np.ndarray) -> NoReturn:
        val_pred = self._forward_prop(val_data)
        self.metrics(val_label, val_pred)

    def predict(self, test_data: np.ndarray) -> np.ndarray:
        return self._forward_prop(test_data)


if __name__ == '__main__':
    from static.layers import Linear,Activation
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    from static.activations import tanh, sigmoid
    from static.optimizers import SGD
    from static.loss import MSE
    from sklearn.preprocessing import normalize

    # load data
    boston = load_boston()
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2, random_state=0)
    x_train = normalize(x_train, axis=0)
    y_train = normalize(np.expand_dims(y_train, axis=1), axis=0)

    # build model
    model = SequentialNet()
    model.add(Linear(input_dim=13, output_dim=4))
    model.add(Activation(tanh()))
    model.add(Linear(input_dim=4, output_dim=1))
    model.add(Activation(sigmoid()))
    model.compile(optimizer=SGD(lr=0.001), loss=MSE )
    model.show()

    # train and eval
    model.fit(train_data=x_train, train_label=y_train, epochs=20)