#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: ch06_rnn.py
Date: 2019/8/21 2:51 PM
"""
from mxnet import nd, init, gluon
from mxnet.gluon import nn, rnn, loss as gloss
from common import data_sets, log_utils
from common.base_rnn import BaseRNN


logger = log_utils.get_logger(__name__)


class RNNScratch(BaseRNN):
    """
    """
    def __init__(self, vocab_size, idx_to_char, char_to_idx, num_hidden, **kwargs):
        super(RNNScratch, self).__init__(vocab_size, idx_to_char, char_to_idx, num_hidden, **kwargs)
        # model params
        self.w_xh = None
        self.w_hh = None
        self.b_h = None
        self.w_hq = None
        self.b_q = None
        # loss function
        self.loss = gloss.SoftmaxCrossEntropyLoss()

    def begin_state(self, batch_size):
        """ init first state """
        return nd.zeros(shape=(batch_size, self.num_hidden), ctx=self.ctx),

    def fit(self, corpus_indices, num_steps, hyper_params, epochs, **kwargs):
        """ fit function """
        self.w_xh = self._one_params(shape=(self.vocab_size, self.num_hidden))
        self.w_hh = self._one_params(shape=(self.num_hidden, self.num_hidden))
        self.b_h = nd.zeros(shape=(self.num_hidden,), ctx=self.ctx)
        self.w_hq = self._one_params(shape=(self.num_hidden, self.vocab_size))
        self.b_q = nd.zeros(shape=(self.vocab_size,), ctx=self.ctx)

        self.parameters = [self.w_xh, self.w_hh, self.b_h, self.w_hq, self.b_q]
        for param in self.parameters:
            param.attach_grad()

        is_random_iter = kwargs.get("is_random_iter", False)
        self._train(corpus_indices, num_steps, hyper_params, epochs, is_random_iter)

    def forward(self, inputs, state):
        """
        forward function
        inputs (num_steps, batch_size, vocab_size)
        outputs (num_steps * batch_size, vocab_size)
        """
        h, = state
        outputs = []
        for x in inputs:
            h = nd.tanh(nd.dot(x, self.w_xh) + nd.dot(h, self.w_hh) + self.b_h)
            y = nd.dot(h, self.w_hq) + self.b_q
            outputs.append(y)
        y_hat = nd.concat(*outputs, dim=0)
        return y_hat, (h,)

    def _one_params(self, shape):
        return nd.random.normal(scale=self.init_scale, shape=shape, ctx=self.ctx)


class RNNGluon(BaseRNN):
    """
    """
    def __init__(self, vocab_size, idx_to_char, char_to_idx, num_hidden, **kwargs):
        super(RNNGluon, self).__init__(vocab_size, idx_to_char, char_to_idx, num_hidden, **kwargs)
        # model params
        self.rnn_layer = rnn.RNN(num_hidden)
        self.dense = nn.Dense(self.vocab_size)
        # loss function
        self.loss = gloss.SoftmaxCrossEntropyLoss()

    def begin_state(self, batch_size):
        """init first state"""
        return self.rnn_layer.begin_state(batch_size=batch_size, ctx=self.ctx)

    def fit(self, corpus_indices, num_steps, hyper_params, epochs, **kwargs):
        """ fit function """
        lr = hyper_params.get("lr", 1e2)
        self.initialize(ctx=self.ctx, force_reinit=True, init=init.Normal(sigma=0.01))
        self.trainer = gluon.Trainer(self.collect_params(), "sgd",
                                     {"learning_rate": lr, "momentum": 0, "wd": 0})
        self._train(corpus_indices, num_steps, hyper_params, epochs, False)

    def forward(self, inputs, state):
        """ forward function """
        h, state = self.rnn_layer(inputs, state)
        outputs = self.dense(h.reshape((-1, h.shape[-1])))
        return outputs, state


if __name__ == '__main__':
    _corpus_indices, _idx_to_char, _char_to_idx, _vocab_size = \
        data_sets.load_jaychou_lyrics("../data/jaychou_lyrics.txt.zip")
    _num_hidden = 256
    _num_steps = 35
    _batch_size = 32
    _lr = 1e2
    use_gluon = True
    if use_gluon:
        model = RNNGluon(_vocab_size, _idx_to_char, _char_to_idx, _num_hidden)
    else:
        model = RNNScratch(_vocab_size, _idx_to_char, _char_to_idx, _num_hidden)

    model.fit(_corpus_indices, _num_steps, {"lr": _lr, "batch_size": _batch_size}, epochs=250)

