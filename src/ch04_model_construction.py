#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: ch04_model_construction.py
Date: 2019/8/20 3:49 PM
"""
from mxnet import nd
from mxnet.gluon import nn
from common import log_utils

import codecs


logger = log_utils.get_logger(__name__)


class MLP(nn.Block):
    """
    """
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation="relu")
        self.output = nn.Dense(10)

    def forward(self, x):
        return self.output(self.hidden(x))


class MySequential(nn.Block):
    """
    """
    def __init__(self, **kwargs):
        super(MySequential, self).__init__(**kwargs)

    def add(self, block):
        """ add block """
        self._children[block.name] = block

    def forward(self, x):
        """forward function """
        for block in self._children.values():
            x = block(x)
        return x


class FancyMLP(nn.Block):
    """
    """
    def __init__(self, **kwargs):
        super(FancyMLP, self).__init__(**kwargs)
        self.rand_weight = self.params.get_constant("rand_weight", nd.random.uniform(shape=(20, 20)))
        self.dense = nn.Dense(20, activation="relu")

    def forward(self, x):
        """ forward function """
        x = self.dense(x)
        x = nd.relu(nd.dot(x, self.rand_weight.data()) + 1)
        x = self.dense(x)
        while x.norm().asscalar() > 1:
            x /= 2

        if x.norm().asscalar() < 0.8:
            x *= 10
        return x.sum()


class NestMLP(nn.Block):
    """
    """
    def __init__(self, **kwargs):
        super(NestMLP, self).__init__(**kwargs)
        self.net = nn.Sequential()
        self.net.add(nn.Dense(64, activation="relu"),
                     nn.Dense(32, activation="relu"))
        self.dense = nn.Dense(16, activation="relu")

    def forward(self, x):
        """ forward function """
        return self.dense(self.net(x))


if __name__ == '__main__':
    X = nd.random.uniform(shape=(2, 20))
    net = MLP()
    net.initialize()
    logger.info("\n############# MLP result ##################")
    logger.info(net(X))

    net = MySequential()
    net.add(nn.Dense(256, activation="relu"))
    net.add(nn.Dense(10))
    net.initialize()
    logger.info("\n############# Sequential result ##################")
    logger.info(net(X))

    net = FancyMLP()
    net.initialize()
    logger.info("\n############# FancyMLP result ##################")
    logger.info(net(X))

    net = nn.Sequential()
    net.add(NestMLP(), nn.Dense(20), FancyMLP())
    net.initialize()
    logger.info("\n############# NestMLP result ##################")
    logger.info(net(X))

