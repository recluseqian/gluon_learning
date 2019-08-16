#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: functions.py
Date: 2019/8/16 8:44 PM
"""
from common import log_utils


def sgd(params, lr, batch_size):
    """
    """
    for param in params:
        param -= lr / batch_size * param.grad


def square_loss(y_hat, y):
    """
    """
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
