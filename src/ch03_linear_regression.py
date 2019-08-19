#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: ch03_linear_regression.py
Date: 2019/8/6 11:05 AM
"""
from mxnet import gluon, nd, init
from mxnet.gluon import nn, loss as g_loss, data as g_data

from common.base_model import BaseRegression
from common import data_sets
from common import log_utils
from common.functions import square_loss


logger = log_utils.get_logger(__name__)


class LinRegScratch(BaseRegression):
    """
    """

    def __init__(self, **kwargs):
        super(LinRegScratch, self).__init__(**kwargs)
        self.w = None
        self.b = None
        if not self.loss:
            self.loss = square_loss

    def fit(self, train_x, train_y, lr=0.3, batch_size=64, epochs=3,
            test_x=None, test_y=None):
        """
        """
        num_examples, num_features = train_x.shape
        self.w = nd.random.normal(scale=1, shape=(num_features, 1))
        self.b = nd.zeros(shape=(1,))
        self.w.attach_grad()
        self.b.attach_grad()
        self.params.append(self.w)
        self.params.append(self.b)

        for epoch in range(epochs):
            for x, y in data_sets.iter_data(train_x, train_y, batch_size):
                self._train(x, y, lr, batch_size)
            train_loss = self.evaluate(train_x, train_y)
            if test_x and test_y:
                test_loss = self.evaluate(test_x, test_y)
                logger.info("epoch {} - train_loss: {}, test_loss: {}".format(epoch + 1, train_loss, test_loss))
            else:
                logger.info("epoch {} - train_loss: {}".format(epoch + 1, train_loss))

    def forward(self, x):
        """
        """
        y_hat = nd.dot(x, self.w) + self.b
        return y_hat

    def print_model(self):
        """
        """
        logger.info("learned w: {}".format(self.w))
        logger.info("learned b: {}".format(self.b))


class LinRegGluon(BaseRegression):
    """
    """
    def __init__(self):
        super(LinRegGluon, self).__init__()
        self.net = None
        if not self.loss:
            self.loss = g_loss.L2Loss()

    def fit(self, train_x, train_y, lr=0.03, batch_size=64, epochs=10,
            test_x=None, test_y=None):
        """
        """
        # data set
        data_set = g_data.ArrayDataset(train_x, train_y)
        data_iter = g_data.DataLoader(data_set, batch_size)

        # model
        self.net = nn.Sequential()
        self.net.add(nn.Dense(1))
        self.net.initialize(init.Normal(sigma=0.01))

        # training
        self.trainers.append(gluon.Trainer(self.net.collect_params(), 'sgd', {"learning_rate": lr}))
        for epoch in range(epochs):
            for x, y in data_iter:
                self._train(x, y, lr, batch_size)

            train_loss = self.evaluate(train_x, train_y)
            if test_x and test_y:
                test_loss = self.evaluate(test_x, test_y)
                logger.info("epoch {} - train_loss: {}, test_loss: {}".format(epoch + 1, train_loss, test_loss))
            else:
                logger.info("epoch {} - train_loss: {}".format(epoch + 1, train_loss))

    def forward(self, x):
        """
        """
        return self.net(x)

    def print_model(self):
        """
        """
        dense = self.net[0]
        logger.info("learned w: {}".format(dense.weight.data()))
        logger.info("learned b: {}".format(dense.bias.data()))


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--use_gluon", default="1")
    args = parser.parse_args()
    # load data
    true_w = [2, -3.4, 4]
    true_b = 4.2
    _x, _y = data_sets.load_data_linear_regression(true_w, true_b)
    # model
    if args.use_gluon == "1":
        model = LinRegGluon()
    else:
        model = LinRegScratch()
    # training
    model.fit(_x, _y, epochs=30)

    model.print_model()
