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


class PolynomialRegression(BaseRegression):
    """
    """

    def __init__(self, **kwargs):
        super(PolynomialRegression, self).__init__(**kwargs)
        self.net = None
        self.loss = gloss.L2Loss()

    def fit(self, train_x, train_y, lr=0.01, batch_size=64, epochs=10,
            test_x=None, test_y=None):
        """ fit x, y"""
        train_data_set = gdata.ArrayDataset(train_x, train_y)
        train_iter = gdata.DataLoader(train_data_set, batch_size=batch_size, shuffle=True)

        self.net = nn.Sequential()
        self.net.add(nn.Dense(1))
        self.net.initialize()

        self.trainer = gluon.Trainer(self.net.collect_params(), "sgd",
                                     {"learning_rate": lr})
        train_loss, test_loss = [], []
        for epoch in range(epochs):
            for x, y in train_iter:
                self._train(x, y, lr, batch_size)

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
    num_train = 5000
    num_test = 1000
    data_x, data_y = data_sets.load_data_polynomial([1.2, -3.4, 5.6], 5,
                                                    num_train, num_test)

    _train_x, _train_y = data_x[:num_train], data_y[:num_train]
    _test_x, _test_y = data_x[-num_test:], data_y[-num_test:]

    model = PolynomialRegression()
    model.fit(_train_x, _train_y, test_x=_test_x, test_y=_test_y)
