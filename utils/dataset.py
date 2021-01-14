#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 1/13/21 3:11 下午
# @Author  : pcw
# @File    : dataset.py
# @Description: <>
import numpy as np
from typing import NoReturn,Tuple,List


def data_split(data: np.ndarray, shuffle: bool=True) -> Tuple[np.ndarray, np.ndarray]:
    pass


def shuffle(x: np.ndarray, y: np.ndarray) -> List[np.ndarray]:
    # 仅当x,y维数相同时
    y_dim = y.shape[1]
    xy = np.hstack((x,y))
    np.random.shuffle(xy)
    return np.hsplit(xy, [-y_dim])


class DatasetBatchIter:
    """ 批样本生成器
    """
    def __init__(self, data_x: np.ndarray, data_y: np.ndarray, batch_size: int=16, copy: bool=True, is_limit: bool=True):
        self.x = np.copy(data_x) if copy else data_x
        self.y = np.copy(data_y) if copy else data_y
        self.batch_size = batch_size
        self.start = -batch_size
        self.n = len(data_x)-1
        self.is_limit = is_limit

    def __iter__(self):
        return self

    def __next__(self):
        self.start += self.batch_size
        if self.start > self.n:
            if self.is_limit:
                raise StopIteration
            else:
                self._shuffle()
                self.start = 0
        return (self.x[self.start: self.start+self.batch_size][:], self.y[self.start: self.start+self.batch_size][:])

    def _shuffle(self):
        self.x, self.y = shuffle(self.x, self.y)


if __name__ == '__main__':
    x = np.random.rand(10, 3)
    y = np.random.randint(2, size=(10, 1))
    ds = DatasetBatchIter(x, y, batch_size=3, is_limit=False)
    i = 0
    while i<10:
        print(next(ds))
        i += 1
