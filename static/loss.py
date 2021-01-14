#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 12/28/20 11:51 下午
# @Author  : pcw
# @File    : loss.py
# @Description: <>
import numpy as np
from typing import NamedTuple, Callable
from static.activations import softmax_func

class Loss(NamedTuple):
    batch_loss: Callable
    loss_grad: Callable


def mse(predict: np.ndarray, target: np.ndarray) -> np.ndarray:
    """均方误差
        mean squared error loss, or L2 loss
    Args:
        predict: (n_samples, 1)
        target: (n_samples, 1)
    """
    return np.sum(np.power(predict - target, 2)) / predict.shape[0]

def mse_grad(predict: np.ndarray, target: np.ndarray) -> np.ndarray:
    """均方误差导数
    """
    return 2 * (predict - target)

MSE = Loss(mse, mse_grad)


def mae(predict: np.ndarray, target: np.ndarray) -> np.ndarray:
    """平均绝对误差
        mean absolute error loss, or L1 loss
    Args:
        predict: (n_samples, 1)
        target: (n_samples, 1)

    Returns:

    """
    return np.sum(np.abs(predict - target)) / predict.shape[0]

_epsilon_ = 1e-5

def binary_cross_entropy(predict: np.ndarray, target: np.ndarray, samples_reduce: bool=True) -> np.ndarray:
    """二元交叉熵
        binary cross entropy loss
        -y_true*log(1-y_pred) - (1-y_true)*log(y_pred))
        = -log(1-y_pred) if y==0 else -log(y_pred)
    Args:
        predict: (n_samples, 1)
        target: (n_samples, 1)
        samples_reduce: 多样本平均后返回
    Returns:

    Notes:
        tf中使用了logistic形式计算,通过变换可以防止数值溢出
            sigmoid_cross_entropy = max(x, 0) - x * z + log(1 + exp(-abs(x)))
        这里仅使用截断的方式防止溢出

    References:
        https://github.com/tensorflow/tensorflow/blob/582c8d236cb079023657287c318ff26adb239002/tensorflow/python/ops/nn_impl.py#L196-L244
        https://github.com/tensorflow/tensorflow/blob/582c8d236cb079023657287c318ff26adb239002/tensorflow/python/keras/backend.py#L4956
    """
    predict = np.clip(predict, _epsilon_, 1 - _epsilon_)
    # use ln() here, can also be np.log2()
    bce = - np.sum(target * np.log(predict) + (1 - target) * np.log(1 - predict), axis=-1)
    #print("bce: \n{}".format(bce))
    return  np.sum(bce) / predict.shape[0] if samples_reduce else bce


def categorical_cross_entropy(predict: np.ndarray, target: np.ndarray, samples_reduce: bool=True) -> np.ndarray:
    """多元交叉熵
        categorical cross entropy loss
    Args:
        predict: (n_samples, k)
        target: (n_samples, k)

    Returns:

    """
    # use ln() here, can also be np.log2()
    predict = np.clip(predict, _epsilon_, 1 - _epsilon_)
    ce = np.sum(- target * np.log(predict), axis=-1)
    return  np.mean(ce, axis=0) if samples_reduce else ce


def softmax_cross_entropy(predict: np.ndarray, target: np.ndarray, samples_reduce: bool=True) -> np.ndarray:
    """softmax+交叉熵 损失结合
    Args:
        predict: (n_samples, k)
        target: (n_samples, k)
    """
    predict_softmax = softmax_func(predict)
    # predict = np.clip(predict, _epsilon_, 1 - _epsilon_)
    ce = np.sum(- target * np.log(predict_softmax), axis=-1)
    return np.mean(ce, axis=0) if samples_reduce else ce


def softmax_cross_entropy_grad(predict: np.ndarray, target: np.ndarray) -> np.ndarray:
    """softmax+交叉熵 损失结合
    Args:
        predict: (n_samples, k)
        target: (n_samples, k)
    Returns:
        out: (n_samples, k)
    References:
        https://zhuanlan.zhihu.com/p/60042105
    """
    return softmax_func(predict) - target


SCE = Loss(softmax_cross_entropy, softmax_cross_entropy_grad)


if __name__ == '__main__':
    pred = np.array([[0.6],[0.4]])
    expect = np.array([[0], [0]])
    print("expect: \n{}, \nget \n{}, \nbin loss: {}".format(expect, pred, binary_cross_entropy(pred, expect)))

    expect= np.array([[0, 1, 0], [0, 0, 1]])
    pred = np.array([[0.05, 0.95, 0.],[0.1, 0.8, 0.1]])
    print("expect: \n{}, \nget \n{}, \ncate loss: {}".format(expect, pred, categorical_cross_entropy(pred, expect)))
