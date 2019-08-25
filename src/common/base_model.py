#!/usr/bin/env python
# -*- coding:utf8 -*-
import time
import math
from mxnet import nd, autograd
from common.functions import sgd, grad_clipping
from . import log_utils, data_sets


logger = log_utils.get_logger(__name__)


class BaseRegression:
    """
    """
    def __init__(self, **kwargs):
        self.params = []
        self.trainers = []
        self.loss = kwargs.get('loss')

    def fit(self, train_x, train_y, optimizer, hyper_params, batch_size, epochs, test_x, test_y):
        """
        """
        raise NotImplementedError

    def forward(self, x):
        """
        """
        raise NotImplementedError

    def evaluate(self, x, y):
        """
        """
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        return loss.mean().asscalar()

    def _train(self, x, y, optimizer, hyper_params):
        """
        """
        with autograd.record():
            y_hat = self.forward(x)
            batch_loss = self._loss(y_hat, y).mean()
        batch_loss.backward()
        if self.trainers:
            [trainer.step(1) for trainer in self.trainers]
        else:
            if optimizer == "sgd":
                sgd(self.params, hyper_params)
            else:
                raise ValueError("Only support sgd optimizer")

    def _loss(self, y_hat, y):
        """ loss function """
        return self.loss(y_hat, y)


class BaseClassifier:
    """
    """
    def __init__(self, **kwargs):
        self.params = []
        self.trainers = []
        self.loss = None

    def fit(self, train_iter, hyper_params, batch_size, epochs, test_iter):
        """
        """
        raise NotImplementedError

    def forward(self, *args):
        raise NotImplementedError

    def evaluate_accuracy(self, data_iter):
        """
        """
        acc_num, total_num = 0.0, 0.0
        for x, y in data_iter:
            y = y.astype('float32')
            y_hat = self.forward(x)
            acc_num += (y_hat.argmax(axis=1) == y).sum().asscalar()
            total_num += y.size
        return acc_num / total_num

    def _train(self, train_iter, hyper_params, batch_size, epochs, test_iter=None):
        """
        """
        for epoch in range(epochs):
            train_loss, train_total, train_acc = 0.0, 0, 0.0
            for x, y in train_iter:
                with autograd.record():
                    y_hat = self.forward(x)
                    batch_loss = self._loss(y_hat, y).mean()
                batch_loss.backward()

                if self.trainers:
                    [trainer.step(1) for trainer in self.trainers]
                else:
                    sgd(self.params, hyper_params)

                train_loss += batch_loss.asscalar()
                y = y.astype("float32")
                train_acc += (y_hat.argmax(axis=1) == y).sum().asscalar()
                train_total += y.size

            train_loss /= train_total
            train_acc /= train_total

            if test_iter:
                test_acc = self.evaluate_accuracy(test_iter)
                logger.info("epoch %d, loss: %.4f, train acc %.4f, test acc: %.4f"
                            % (epoch + 1, train_loss, train_acc, test_acc))

    def _loss(self, y_hat, y):
        """ loss function """
        return self.loss(y_hat, y)


if __name__ == '__main__':
    pass
