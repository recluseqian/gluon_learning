#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: ch03_polynomial_regression.py
Date: 2019/8/18 11:07 AM
"""
from mxnet import nd, gluon, init
from mxnet.gluon import data as gdata, nn, loss as gloss
from common import data_sets, log_utils, display_utils
from common.base_model import BaseRegression


logger = log_utils.get_logger(__name__)


class PolynomialRegressionGluon(BaseRegression):
    """
    """

    def __init__(self, **kwargs):
        super(PolynomialRegressionGluon, self).__init__(**kwargs)
        self.net = None
        self.loss = gloss.L2Loss()
        self.regularization = kwargs.get("regularization")

    def fit(self, train_x, train_y, optimizer, hyper_params, batch_size=64, epochs=10,
            test_x=None, test_y=None):
        """ fit x, y"""
        train_data_set = gdata.ArrayDataset(train_x, train_y)
        train_iter = gdata.DataLoader(train_data_set, batch_size=batch_size, shuffle=True)

        self.net = nn.Sequential()
        self.net.add(nn.Dense(1))
        self.net.initialize()

        if optimizer != "sgd":
            raise ValueError("only support sgd optimizer")

        if self.regularization == "l2":
            weight_decay_params = {k: v for k, v in hyper_params}
            weight_decay_params["wd"] = 3
            self.trainers = [
                gluon.Trainer(self.net.collect_params(".*weight"), "sgd", weight_decay_params),
                gluon.Trainer(self.net.collect_params(".*bias"), "sgd", hyper_params)

            ]
        else:
            self.trainers = [
                gluon.Trainer(self.net.collect_params(), "sgd", hyper_params)
            ]

        train_loss, test_loss = [], []
        for epoch in range(epochs):
            for x, y in train_iter:
                self._train(x, y, optimizer, hyper_params)

            train_loss.append(self.evaluate(train_x, train_y))
            test_loss.append(self.evaluate(test_x, test_y))
        logger.info("final epoch train loss: {}, test loss: {}"
                    .format(train_loss[-1], test_loss[-1]))
        display_utils.semilogy(range(1, epochs + 1), train_loss, 'epochs', 'loss',
                               x2_vals=range(1, epochs + 1), y2_vals=test_loss,
                               legend=["train", "test"])

        logger.info("weight: {}\nbias: {}".format(self.net[0].weight.data().asnumpy(),
                                                  self.net[0].bias.data().asnumpy()))

    def forward(self, x):
        """ forward propagation """
        return self.net(x)


if __name__ == '__main__':
    pass
