#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: linear_regression.py
Date: 2019/8/6 11:05 AM
"""
import sys
from mxnet import nd, autograd
sys.path.append("../")
from utils import data_utils
from utils import log_utils


logger = log_utils.get_logger(__name__)


def sgd(params, lr, batch_size):
    """
    """
    for param in params:
        param[:] = param - lr / batch_size * param.grad


class LinearRegression:
    """
    """
    def __init__(self):
        self.w = None
        self.b = None
        self.num_features = 0
        self.num_examples = 0

    def fit(self, x, y, batch_size=64, epochs=3):
        """
        """
        self.num_examples, self.num_features = x.shape
        self.w = nd.random.normal(scale=1, shape=(self.num_features, 1))
        self.b = nd.zeros(shape=(1,))
        self.w.attach_grad()
        self.b.attach_grad()

        for epoch in range(epochs):
            for batch_x, batch_y in data_utils.iter_data(x, y, batch_size):
                with autograd.record():
                    epoch_loss = self.loss(x, y)
                epoch_loss.backward()
                sgd([self.w, self.b], lr=0.03, batch_size=batch_size)
            train_loss = self.loss(x, y)
            logger.info("$poch {}, loss {}".format(epoch + 1, train_loss
                                                   .mean().asnumpy()))

    def feed_forward(self, x):
        """
        """
        y_hat = nd.dot(x, self.w) + self.b
        return y_hat

    def loss(self, x, y):
        """
        """
        y_hat = self.feed_forward(x)
        squared_loss = (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
        return squared_loss


if __name__ == "__main__":
    model = LinearRegression()
    x, y = data_utils.load_data()
    model.fit(x, y)
    logger.debug(model.w)
    logger.debug(model.b)
