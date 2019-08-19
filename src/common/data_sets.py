#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: data_sets.py
Date: 2019/8/16 8:45 PM
"""
import sys
from mxnet import nd
from mxnet.gluon import data as gdata
import random

from common import log_utils

logger = log_utils.get_logger(__name__)


def iter_data(x, y, batch_size):
    """
    """
    num_examples = len(x)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = nd.array(indices[i:min(i + batch_size, num_examples)])
        yield x.take(batch_indices), y.take(batch_indices)


def load_data_linear_regression(true_w, true_b, num_train=1000, num_test=0):
    """
    """
    assert isinstance(true_w, list)
    assert isinstance(true_b, float)
    num_features = len(true_w)

    true_w = nd.array(true_w)
    true_b = nd.array([true_b, ])

    x = nd.random.normal(scale=1, shape=(num_train + num_test,
                                         num_features))
    y = nd.dot(x, true_w) + true_b
    y += nd.random.normal(scale=0.01, shape=y.shape)
    return x, y


def load_data_fashion_mnist(batch_size=256):
    """
    """
    mnist_train = gdata.vision.FashionMNIST(train=True)
    mnist_test = gdata.vision.FashionMNIST(train=False)
    transformer = gdata.vision.transforms.ToTensor()
    if sys.platform.startswith("win"):
        num_workers = 0
    else:
        num_workers = 4

    train_iter = gdata.DataLoader(mnist_train.transform_first(transformer),
                                  batch_size, shuffle=True,
                                  num_workers=num_workers)
    test_iter = gdata.DataLoader(mnist_test.transform_first(transformer),
                                 batch_size, shuffle=False,
                                 num_workers=num_workers)
    return train_iter, test_iter


def get_fashion_mnist_labels(labels):
    """
    """
    text_labels = ["t-shirt", "trouser", "pullover", "dress", "cost",
                   "sandal", "shirt", "sneaker", "bag", "ankle_boot"]
    return [text_labels[int(i)] for i in labels]


def load_data_polynomial(true_w, true_b, num_train=5000, num_test=1000):
    """
    """
    features = nd.normal(shape=(num_train + num_test, 1))
    poly_features = [nd.power(features, i) for i in range(1, len(true_w) + 1)]
    poly_features = nd.concat(*poly_features)
    labels = nd.dot(poly_features, true_w) + true_b
    labels += nd.random.normal(scale=0.1)
    return features, poly_features, labels


if __name__ == '__main__':
    # test fashion minist
    # import time
    # _train_iter, _test_iter = load_data_fashion_mnist()
    # start = time.time()
    # for x, y in _train_iter:
    #     continue

    # print("%.2f" % (time.time() - start))
    x, y = load_data_polynomial([1, 2], 1, num_train=1, num_test=1)
    print(x)
    print(y)

