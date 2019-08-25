#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: functions.py
Date: 2019/8/16 8:44 PM
"""
import mxnet as mx
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


# ############ regularization ##################
def l2_penalty(w):
    """ l2 regularization """
    return (w ** 2).sum() / 2


# ############ optimization algorithm ##################
def sgd(params, hyper_params):
    """
    """
    for param in params:
        param -= hyper_params['learning_rate'] * param.grad


def sgd_momentum(params, states, hyper_params):
    """ momentum sgd """
    for p, v in zip(params, states):
        v[:] = hyper_params["momentum"] * v + hyper_params["learning_rate"] * p.grad
        p[:] -= v


def adagrad(params, states, hyper_params):
    """ ada grad optimizer """
    eps = 1e-6
    for p, s in zip(params, states):
        s[:] += p.grad.square()
        p[:] -= hyper_params["learning_rate"] * p.grad / (s + eps).sqrt()


def rmsprop(params, states, hyper_params):
    """ RMSProp optimizer """
    gamma1, eps = hyper_params["gamma1"], 1e-6
    for p, s in zip(params, states):
        s[:] += gamma1 * s + (1 - gamma1) * p.grad.square()
        p[:] -= hyper_params["learning_rate"] * p.grad / (s + eps).sqrt()


def adadelta(params, states, hyper_params):
    """ ada delta """
    rho, eps = hyper_params["rho"], 1e-5
    for p, (s, delta) in zip(params, states):
        s[:] += rho * s + (1 - rho) * p.grad.square()
        g = ((delta + eps).sqrt() / (s + eps).sqrt()) * p.grad
        p[:] -= g
        delta[:] = rho * delta + (1 - rho) * g.square()


def adam(params, states, hyper_params):
    """ adam """
    beta1, beta2, eps = hyper_params.get("beta1", 0.9), hyper_params.get("beta2", 0.999), 1e-6
    for p, (v, s) in zip(params, states):
        v[:] = beta1 * v + (1 - beta1) * p.grad
        s[:] = beta2 * s + (1 - beta2) * p.grad.square()
        v_bias_corr = v / (1 - beta1 ** hyper_params["t"])
        s_bias_corr = s / (1 - beta2 ** hyper_params["t"])
        p[:] -= hyper_params["learning_rate"] * v_bias_corr / (s_bias_corr.sqrt() + eps)


# ############ others #####################
def grad_clipping(params, theta, ctx):
    if theta is None:
        return

    norm = nd.array([0], ctx=ctx)
    for param in params:
        norm += (param.grad ** 2).sum()
    norm = norm.sqrt().asscalar()
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


def try_gpu():
    """ If gpu is available return gpu context, else return cpu context"""
    try:
        ctx = mx.gpu()
        _ = nd.array([0], ctx=ctx)
    except mx.base.MXNetError:
        ctx = mx.cpu()
    return ctx


if __name__ == '__main__':
    # test cross entropy
    _y_hat = nd.array([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
    _y = nd.array([0, 2], dtype='int32')
    logger.info("cross entropy: {}".format(cross_entropy(_y_hat, _y)))
