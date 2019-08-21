#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: ch06_rnn.py
Date: 2019/8/21 2:51 PM
"""
import math
from mxnet import nd, autograd
from mxnet.gluon import nn, loss as gloss
from common import data_sets, log_utils
from common.base_model import BaseClassifier


class RNNScratch(BaseClassifier):
    """
    """
    def __init__(self, **kwargs):
        self.ctx = kwargs.get("context")
        self.num_inputs = int(kwargs["num_inputs"])
        self.num_hidden = int(kwargs["num_hidden"])
        self.num_outputs = int(kwargs["num_outputs"])
        self.w_xh = None
        self.w_hh = None
        self.b_h = None
        self.w_ho = None
        self.b_o = None

    def fit(self, train_iter, lr, batch_size, epochs, test_iter):
        """ fit data """
        def _get_params(shape):
            return nd.random.normal(scale=0.01, shape=shape, ctx=self.ctx)

        self.w_xh = _get_params((self.num_inputs, self.num_hidden))
        self.w_hh = _get_params((self.num_hidden, self.num_hidden))
        self.b_h = nd.zeros(self.num_hidden, ctx=self.ctx)

        self.w_ho = _get_params((self.num_hidden, self.num_outputs))
        self.b_o = nd.zeros(self.num_outputs, ctx=self.ctx)

        self.params = [self.w_xh, self.w_hh, self.b_h, self.w_ho, self.b_o]
        for param in self.params:
            param.attach_grad()

    def forward(self, x):
        """ forward function """


