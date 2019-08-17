#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: ch03_mlp.py
Date: 2019/8/17 6:39 PM
"""
from mxnet import nd, gluon, init
from mxnet.gluon import nn, loss as gloss
from common.data_sets import load_data_fashion_mnist
from common.base_model import BaseClassifier
from common.functions import relu, soft_max, cross_entropy
from common import log_utils


logger = log_utils.get_logger(__name__)


class MLPScratch(BaseClassifier):
    """
    """
    def __init__(self, **kwargs):
        super(MLPScratch, self).__init__(**kwargs)
        self.net_sizes = kwargs["net_sizes"]
        self.input_size = self.net_sizes[0]
        self.weights = None
        self.biases = None

        self.loss = gloss.SoftmaxCrossEntropyLoss()

    def fit(self, train_iter, lr=0.5, batch_size=256, epochs=10, test_iter=None):
        """
        """
        self.weights = [nd.random.normal(scale=0.01, shape=(in_size, out_size))
                        for in_size, out_size in zip(self.net_sizes[:-1], self.net_sizes[1:])]
        self.biases = [nd.zeros(out_size, ) for out_size in self.net_sizes[1:]]
        self.params.extend(self.weights)
        self.params.extend(self.biases)
        for param in self.params:
            param.attach_grad()

        self._train(train_iter, lr, batch_size, epochs, test_iter=test_iter)

    def forward(self, x):
        """
        """
        a = x.reshape((-1, self.input_size))
        for layer, (w, b) in enumerate(zip(self.weights, self.biases), start=1):
            z = nd.dot(a, w) + b
            if layer == len(self.net_sizes):
                a = z
            else:
                a = relu(z)
        return a


class MLPGluon(BaseClassifier):
    """
    """
    def __init__(self, **kwargs):
        super(MLPGluon, self).__init__(**kwargs)
        self.net = None

        self.loss = gloss.SoftmaxCrossEntropyLoss()

    def fit(self, train_iter, lr=0.5, batch_size=256, epochs=10, test_iter=None):
        """
        """
        self.net = nn.Sequential()
        self.net.add(nn.Dense(256, activation='relu'))
        self.net.add(nn.Dense(10))
        self.net.initialize(init.Normal(sigma=0.01))

        self.trainer = gluon.Trainer(self.net.collect_params(), "sgd", {"learning_rate": lr})

        self._train(train_iter, lr, batch_size, epochs, test_iter=test_iter)

    def forward(self, x):
        """
        """
        return self.net(x)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--use_gluon", default="1")
    args = parser.parse_args()

    _train_iter, _test_iter = load_data_fashion_mnist(batch_size=256)

    if args.use_gluon == "1":
        model = MLPGluon()
    else:
        model = MLPScratch(net_sizes=[784, 256, 10])

    model.fit(_train_iter, test_iter=_test_iter)
