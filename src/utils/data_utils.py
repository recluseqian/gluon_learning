#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: data_utils.py
Date: 2019/8/6 11:04 AM
"""
from mxnet import nd
import random


def load_data():
    """
    """
    num_features = 2
    num_examples = 1000
    true_w = nd.array([13, -5.6])
    true_b = nd.array([13.5, ])
    x = nd.random.normal(scale=1, shape=(num_examples,
                                         num_features))
    y = nd.dot(x, true_w) + true_b
    y += nd.random.normal(scale=0.01, shape=y.shape)
    return x, y


def iter_data(x, y, batch_size):
    """
    """
    num_examples = len(x)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = nd.array(indices[i:min(i + batch_size, num_examples)])
        yield x.take(batch_indices), y.take(batch_indices)


if __name__ == "__main__":
    pass
