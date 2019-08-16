#!/usr/bin/env python
# -*- coding:utf8 -*-

from mxnet import autograd
from common.functions import sgd


class BaseModel:
    """
    """
    def __init__(self, **kwargs):
        self.params = []
        self.loss = None
        self.trainer = None

    def fit(self, train_x, train_y, lr=0.03, batch_size=64, epochs=10,
            test_x=None, test_y=None):
        """
        """
        raise NotImplementedError

    def train(self, x, y, lr=0.03, batch_size=64):
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


# noinspection PyAbstractClass
class BaseRegression(BaseModel):
    """
    """
    def __init__(self, **kwargs):
        super(BaseRegression, self).__init__(**kwargs)


# noinspection PyAbstractClass
class BaseClassifier(BaseModel):
    """
    """
    def __init__(self, **kwargs):
        super(BaseClassifier, self).__init__(**kwargs)


