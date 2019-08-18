#!/usr/bin/env python
# -*- coding:utf8 -*-

from mxnet import autograd
from common.functions import sgd
from . import log_utils


logger = log_utils.get_logger(__name__)


class BaseRegression:
    """
    """
    def __init__(self, **kwargs):
        self.params = []
        self.trainer = None
        self.loss = kwargs.get('loss')

    def fit(self, train_x, train_y, lr, batch_size, epochs, test_x, test_y):
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
        return loss.mean().asnumpy()

    def _train(self, x, y, lr, batch_size):
        """
        """
        with autograd.record():
            y_hat = self.forward(x)
            batch_loss = self.loss(y_hat, y)
        batch_loss.backward()
        if self.trainer:
            self.trainer.step(batch_size)
        else:
            sgd(self.params, lr, batch_size)


class BaseClassifier:
    """
    """
    def __init__(self, **kwargs):
        self.params = []
        self.trainer = None
        self.loss = None

    def fit(self, train_iter, lr, batch_size, epochs, test_iter):
        """
        """
        raise NotImplementedError

    def forward(self, x):
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

    def _train(self, train_iter, lr, batch_size, epochs, test_iter=None):
        """
        """

        for epoch in range(epochs):
            train_loss, train_total, train_acc = 0.0, 0, 0.0
            for x, y in train_iter:
                with autograd.record():
                    y_hat = self.forward(x)
                    batch_loss = self.loss(y_hat, y).sum()
                batch_loss.backward()

                if self.trainer:
                    self.trainer.step(batch_size)
                else:
                    sgd(self.params, lr, batch_size)

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


if __name__ == '__main__':
    pass
