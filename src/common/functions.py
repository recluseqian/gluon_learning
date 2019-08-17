#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: functions.py
Date: 2019/8/16 8:44 PM
"""
from mxnet import nd
from common import log_utils


logger = log_utils.get_logger(__name__)


# ############ activation function ##################
def relu(x):
    """
    """
    return nd.maximum(x, 0)


def soft_max(x):
    """
    """
    fit_x = x - x.max(axis=1, keepdims=True)
    x_exp = fit_x.exp()
    partition = x_exp.sum(axis=1, keepdims=True)
    return x_exp / partition


# ############ loss function ##################
def square_loss(y_hat, y):
    """
    """
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def cross_entropy(y_hat, y):
    """
    """
    return -nd.pick(y_hat, y).log()


# ############ learning algorithm ##################
def sgd(params, lr, batch_size):
    """
    """
    for param in params:
        param -= lr / batch_size * param.grad


if __name__ == '__main__':
    # test cross entropy
    _y_hat = nd.array([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
    _y = nd.array([0, 2], dtype='int32')
    logger.info("cross entropy: {}".format(cross_entropy(_y_hat, _y)))
