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
from utils.dataset import DatasetBatchIter
import numpy as np
import math

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
    """序列式静态图"""
    def __init__(self):
        self.layers: List[Layer] = []

    def show(self):
        print("   Sequential Neural Network   ")
        print("-------------------------------")
        for lay in self.layers:
            print(lay)
        print("-------------------------------")

    def add(self, layer: Layer):
        self.layers.append(layer)

    def fit(self, train_data: np.ndarray, train_label: np.ndarray, epochs: int=5, batch_size: int=16):
        print("input size: {}, target size:{}".format(train_data.shape, train_label.shape))
        if len(train_label.shape)==1:
            train_label = np.expand_dims(train_label, axis=1)
        n_samples = train_data.shape[0]
        batch_num = math.ceil(n_samples/batch_size)
        batchIter = DatasetBatchIter(train_data, train_label, batch_size=batch_size, is_limit=False)
        for epoch in range(epochs):
            i, epoch_total_loss, epoch_mean_metric = 0, 0, 0
            print("\nepoch{} {}".format(epoch+1, "-" * 50))
            while i < batch_num:
                batch_data, batch_label = next(batchIter)
                # mini batch forward propagation
                pred = self._forward_prop(batch_data)
                
                # mini batch backward propagation
                batch_mean_loss = self.loss_func.batch_loss(pred, batch_label)
                loss_grad = self.loss_func.loss_grad(pred, batch_label)
                self._backward_prop(loss_grad)

                # record
                epoch_total_loss += batch_mean_loss
                i += 1
                metric_info = ""
                if self.metrics:
                    batch_metric = self.metrics(batch_label, pred)
                    metric_info = ", metric: {}".format(batch_metric)
                    epoch_mean_metric += batch_metric
                #self.show_model_param()
                print("  batch{}  batch_average_loss: {}{}".format(i, batch_mean_loss, metric_info))
            print("epoch_average_loss: {}".format(epoch_total_loss / batch_num))
            if self.metrics:
                print("epoch_average_metric: {}".format(epoch_mean_metric / batch_num))


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

    def show_model_param(self):
        print("Model Params: ")
        cnt = 1
        for lay in self.layers:
            p = lay.params_to_update if isinstance(lay, ParamLayer) else "Non Params"
            print("Layer{}-{}: {}".format(cnt, lay.layer_name, p))
            cnt += 1
