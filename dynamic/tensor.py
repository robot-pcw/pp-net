#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 12/28/20 11:51 下午
# @Author  : pcw
# @File    : tensor.py
# @Description: <>
from typing import NoReturn
from utils.wraps import ArrayLike, array_like_format

_epsilon_ = 5e-5

def _auto_diff(func, x):
    """有限差分近似
    """
    return (func(x+_epsilon_)-func(x-_epsilon_))/(2*_epsilon_)


class Tensor:
    """numpy array with dynamic grad
    """
    def __init__(self, data: ArrayLike, is_constant: bool=False, name: str="Tensor") -> NoReturn:
        self.data = array_like_format(data)
        self.is_constant = is_constant
        self.name = name

    def __repr__(self):
        return "{} ({}): {}".format(self.name,
                                    "constant" if self.is_constant else "variable",
                                    self.data.shape)


