#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 1/13/21 11:21 上午
# @Author  : pcw
# @File    : metrics.py
# @Description: <>
import numpy as np
from sklearn.metrics import accuracy_score


def acc(true, pred):
    t = np.argmax(true, axis=1)
    p = np.argmax(pred, axis=1)
    return accuracy_score(t, p)