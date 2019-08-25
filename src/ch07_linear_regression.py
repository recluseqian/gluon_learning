#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: ch07_linear_regression.py
Date: 2019/8/25 5:38 PM
"""
from mxnet import gluon, nd, init, autograd
from mxnet.gluon import nn, loss as g_loss, data as g_data

from common.base_model import BaseRegression
from common import log_utils
from common.data_sets import iter_data, load_airfoil_data
from common.functions import square_loss, sgd, sgd_momentum, adagrad, rmsprop, adadelta, adam

logger = log_utils.get_logger(__name__)


class LinRegScratch(BaseRegression):
    """
    """

    def __init__(self, **kwargs):
        super(LinRegScratch, self).__init__(**kwargs)
        self.w = None
        self.b = None
        self.loss = square_loss

    def fit(self, train_x, train_y, optimizer, hyper_params, batch_size=64, epochs=3,
            test_x=None, test_y=None):
        """
        """
        num_examples, num_features = train_x.shape
        self.w = nd.random.normal(scale=1, shape=(num_features, 1))
        self.b = nd.zeros(shape=(1,))
        self.w.attach_grad()
        self.b.attach_grad()
        self.params = [self.w, self.b]

        history = []
        for epoch in range(epochs):
            for x, y in iter_data(train_x, train_y, batch_size):
                self._train(x, y, optimizer, hyper_params)
            train_loss = self.evaluate(train_x, train_y)
            if test_x and test_y:
                test_loss = self.evaluate(test_x, test_y)
                logger.info("epoch {} - train_loss: {}, test_loss: {}".format(epoch + 1, train_loss, test_loss))
            else:
                test_loss = None
                logger.info("epoch {} - train_loss: {}".format(epoch + 1, train_loss))
            history.append((train_loss, test_loss))
        return history

    def _train(self, x, y, optimizer, hyper_params):
        with autograd.record():
            y_hat = self.forward(x)
            batch_loss = self._loss(y_hat, y).mean()
        batch_loss.backward()
        if self.trainers:
            [trainer.step(1) for trainer in self.trainers]
        else:
            if optimizer == "sgd":
                sgd(self.params, hyper_params)
            elif optimizer == "momentum":
                momentum_state = self.init_one_tuple_states()
                sgd_momentum(self.params, momentum_state, hyper_params)
            elif optimizer == "adagrad":
                ada_grad_state = self.init_one_tuple_states()
                adagrad(self.params, ada_grad_state, hyper_params)
            elif optimizer == "rmsprop":
                rmsprop_state = self.init_one_tuple_states()
                rmsprop(self.params, rmsprop_state, hyper_params)
            elif optimizer == "adadelta":
                adadelta_states = self.init_two_tuple_states()
                adadelta(self.params, adadelta_states, hyper_params)
            elif optimizer == "adam":
                adam_states = self.init_two_tuple_states()
                adam(self.params, adam_states, hyper_params)
            else:
                raise ValueError("Do not support {} optimizer".format(optimizer))

    def forward(self, x):
        """
        """
        y_hat = nd.dot(x, self.w) + self.b
        return y_hat

    def init_one_tuple_states(self):
        return nd.zeros(self.w.shape), nd.zeros(self.b.shape)

    def init_two_tuple_states(self):
        return (nd.zeros(self.w.shape), nd.zeros(self.w.shape)), \
               (nd.zeros(self.b.shape), nd.zeros(self.b.shape))


class LinRegGluon(BaseRegression):
    """
    """

    def __init__(self):
        super(LinRegGluon, self).__init__()
        self.net = None
        if not self.loss:
            self.loss = g_loss.L2Loss()

    def fit(self, train_x, train_y, optimizer, hyper_params, batch_size=64, epochs=10,
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
        if optimizer in ("sgd", "momentum"):
            self.trainers = [gluon.Trainer(self.net.collect_params(), 'sgd', hyper_params)]
        elif optimizer in ("adagrad", "rmsprop", "adadelta", "adam"):
            self.trainers = [gluon.Trainer(self.net.collect_params(), optimizer, hyper_params)]
        else:
            raise ValueError("Do not support {} optimizer".format(optimizer))
        history = []
        for epoch in range(epochs):
            for x, y in data_iter:
                self._train(x, y, optimizer, hyper_params)

            train_loss = self.evaluate(train_x, train_y)
            if test_x and test_y:
                test_loss = self.evaluate(test_x, test_y)
                logger.info("epoch {} - train_loss: {}, test_loss: {}".format(epoch + 1, train_loss, test_loss))
            else:
                test_loss = None
                logger.info("epoch {} - train_loss: {}".format(epoch + 1, train_loss))
            history.append((train_loss, test_loss))
        return history

    def forward(self, x):
        """
        """
        return self.net(x)


if __name__ == '__main__':
    features, labels = load_airfoil_data()
    features = features[:1500]
    labels = labels[:1500]

    use_gluon = False
    if use_gluon:
        model = LinRegGluon()
    else:
        model = LinRegScratch()

    _optimizer = "momentum"
    _hyper_params = {"learning_rate": 0.03, "momentum": 0.5}

    model.fit(features, labels, "sgd", _hyper_params, batch_size=100, epochs=10)
    model.fit(features, labels, "momentum", _hyper_params, batch_size=100, epochs=10)
