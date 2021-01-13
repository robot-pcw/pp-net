#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 1/12/21 8:17 下午
# @Author  : pcw
# @File    : ops.py
# @Description: <>
import numpy as np

def one_hot(x: list, num_class: int) -> np.ndarray:
    """将标签转换为one-hot向量

    Args:
        x: 分类标签列表
        num_class: 总类别数

    """
    return np.eye(num_class)[x]


if __name__ == '__main__':
    a = one_hot([0,1,2,1,1,0], 3)
    print(a)
