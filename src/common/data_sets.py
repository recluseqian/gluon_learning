#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: data_sets.py
Date: 2019/8/16 8:45 PM
"""
import sys
import zipfile
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


def load_jaychou_lyrics(zip_file):
    """ """
    with zipfile.ZipFile(zip_file) as zin:
        with zin.open("jaychou_lyrics.txt") as f:
            corpus_chars = f.read().decode("utf8")
    corpus_chars = corpus_chars.replace("\n", " ").replace("\r", " ")
    corpus_chars = corpus_chars[:10000]
    idx_to_char = list(set(corpus_chars))
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    vocab_size = len(char_to_idx)
    corpus_indices = [char_to_idx[char] for char in corpus_chars]
    return corpus_indices, idx_to_char, char_to_idx, vocab_size


def data_iter_random(corpus_indices, batch_size, num_steps, ctx=None):
    """ random sample sequence """
    num_examples = (len(corpus_indices) - 1) // num_steps
    epoch_size = num_examples // batch_size
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)

    def _data(pos):
        return corpus_indices[pos: pos + num_steps]

    for i in range(epoch_size):
        i = i * batch_size
        batch_indices = example_indices[i: i + batch_size]
        x = [_data(j * num_steps) for j in batch_indices]
        y = [_data(j * num_steps + 1) for j in batch_indices]
        yield nd.array(x, ctx), nd.array(y, ctx)


def data_iter_consecutive(corpus_indices, batch_size, num_steps, ctx=None):
    """ consecutive sample sequence """
    corpus_indices = nd.array(corpus_indices, ctx=ctx)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size
    indices = corpus_indices[:batch_size*batch_len].reshape((batch_size, batch_len))

    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        i = i * num_steps
        x = indices[:, i: i + num_steps]
        y = indices[:, i + 1: i + num_steps + 1]
        yield x, y


def to_onehot(inputs, voc_size):
    """ (batch_size, num_step) -> (num_step, (batch_size, voc_size)"""
    return [nd.one_hot(x, voc_size) for x in inputs.T]


if __name__ == '__main__':
    # test fashion minist
    # import time
    # _train_iter, _test_iter = load_data_fashion_mnist()
    # start = time.time()
    # for x, y in _train_iter:
    #     continue

    # print("%.2f" % (time.time() - start))
    # load_jaychou_lyrics()
    my_seq = list(range(30))
    for X, Y in data_iter_random(my_seq, batch_size=2, num_steps=6):
        logger.info("X: {}\nY: {}".format(X, Y))
    my_seq = list(range(30))
    for X, Y in data_iter_consecutive(my_seq, batch_size=2, num_steps=6):
        logger.info("X: {}\nY: {}".format(X, Y))

    X = nd.arange(10).reshape((2, 5))
    X = to_onehot(X, 1027)
    logger.info("{}, {}".format(len(X), X[0].shape))


