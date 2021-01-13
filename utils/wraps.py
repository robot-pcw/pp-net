#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 1/11/21 3:06 下午
# @Author  : pcw
# @File    : wraps.py
# @Description: <>
import numpy as np
from functools import  wraps
from typing import Union


def matrix_assert(k_mats, n_dims):
    """输入维度检查，主要用于矩阵加减法、点乘等计算前的维度检查

    Args:
        k_mats: 需要检查的输入矩阵数量
        n_dims: 输入矩阵的预期维度

    Returns:

    """
    def decorate(func):
        @wraps(func)
        def check(*args, **kwargs):
            for i in range(k_mats):
                m_dim = len(args[i].shape)
                if m_dim != n_dims:
                    raise ValueError("except {}-dims, found {}-dims in matrix{}".format(n_dims, m_dim, i))
            return func(*args, **kwargs)
        return check
    return decorate


ArrayLike = Union[float, list, np.ndarray]


def array_like_format(mat: ArrayLike) -> np.ndarray:
    assert type(mat) in (float, list, np.ndarray), mat
    if isinstance(mat, np.ndarray):
        return mat
    else:
        return np.array(mat)



def array_like_wrap():

    pass

if __name__ == '__main__':
    print(array_like_format(123))