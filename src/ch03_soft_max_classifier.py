#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: ch03_soft_max_classifier.py
Date: 2019/8/17 4:47 PM
"""
from mxnet import nd, gluon, init
from mxnet.gluon import nn, loss as g_loss
from data_sets import load_data_fashion_mnist
from common.base_model import BaseClassifier
from common.functions import cross_entropy, soft_max


class SoftMaxScratch(BaseClassifier):

    def __init__(self, **kwargs):
        super(SoftMaxScratch, self).__init__(**kwargs)
        self.num_inputs = kwargs["num_inputs"]
        self.num_outputs = kwargs["num_outputs"]
        self.w = None
        self.b = None

        self.loss = cross_entropy

    def fit(self, train_iter, hyper_params, batch_size=256, epochs=10, test_iter=None):
        """
        """
        # model init params
        self.w = nd.random.normal(scale=0.01, shape=(self.num_inputs, self.num_outputs))
        self.b = nd.zeros(self.num_outputs)
        self.params = [self.w, self.b]

        for param in self.params:
            param.attach_grad()

        # training
        self._train(train_iter, hyper_params, batch_size, epochs, test_iter=test_iter)

    def forward(self, x):
        """
        """
        return soft_max(nd.dot(x.reshape(-1, self.num_inputs), self.w) + self.b)


class SoftMaxGluon(BaseClassifier):
    """
    """
    def __init__(self, **kwargs):
        super(SoftMaxGluon, self).__init__(**kwargs)
        self.net = None
        self.loss = g_loss.SoftmaxCrossEntropyLoss()

    def fit(self, train_iter, hyper_params, batch_size=256, epochs=10, test_iter=None):
        """
        """
        # model initialize
        self.net = nn.Sequential()
        self.net.add(nn.Dense(10))
        self.net.initialize(init.Normal(sigma=0.01))

        # trainer
        self.trainers.append(gluon.Trainer(self.net.collect_params(), "sgd", hyper_params))
        self._train(train_iter, hyper_params, batch_size, epochs, test_iter=test_iter)

    def forward(self, x):
        """
        """
        return self.net(x)


if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--use_gluon", default="0")
    args = parser.parse_args()

    _train_iter, _test_iter = load_data_fashion_mnist(batch_size=256)
    user_gluon = True
    if args.use_gluon == "1":
        model = SoftMaxGluon()
    else:
        model = SoftMaxScratch(num_inputs=784, num_outputs=10)
    model.fit(_train_iter, hyper_params={"learning_rate": 0.3}, test_iter=_test_iter)
