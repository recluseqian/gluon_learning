#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: utils.py
Date: 2019/8/21 8:03 PM
"""
import mxnet as mx
from mxnet import nd


def try_gpu():
    """ If gpu is available return gpu context, else return cpu context"""
    try:
        ctx = mx.gpu()
        _ = nd.array([0], ctx=ctx)
    except mx.base.MXNetError:
        ctx = mx.cpu()
    return ctx
