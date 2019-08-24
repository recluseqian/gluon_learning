#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: ch06_gru.py
Date: 2019/8/24 5:09 PM
"""
from mxnet import nd
from mxnet.gluon import rnn, loss as gloss
from common.base_rnn import BaseRNNScratch, BaseRNNGluon
from common.data_sets import load_jaychou_lyrics


class GRUScratch(BaseRNNScratch):
    """
    """
    def __init__(self, vocab_size, idx_to_char, char_to_idx, num_hidden, **kwargs):
        super(GRUScratch, self).__init__(vocab_size, idx_to_char, char_to_idx, num_hidden, **kwargs)
        # model params
        self.w_xr, self.w_hr, self.b_r = None, None, None
        self.w_xz, self.w_hz, self.b_z = None, None, None
        self.w_xh, self.w_hh, self.b_h = None, None, None
        self.w_hq, self.b_q = None, None

    def begin_state(self, batch_size):
        """ init first state """
        return nd.zeros(shape=(batch_size, self.num_hidden), ctx=self.ctx),

    def fit(self, corpus_indices, num_steps, hyper_params, epochs, **kwargs):
        """ fit function """
        self.w_xr, self.w_hr, self.b_r = self._three_params()
        self.w_xz, self.w_hz, self.b_z = self._three_params()
        self.w_xh, self.w_hh, self.b_h = self._three_params()
        self.w_hq = self._one_params(shape=(self.num_hidden, self.vocab_size))
        self.b_q = nd.zeros(self.vocab_size, ctx=self.ctx)

        self.parameters = [self.w_xr, self.w_hr, self.b_r,
                           self.w_xz, self.w_hz, self.b_z,
                           self.w_xh, self.w_hh, self.b_h,
                           self.w_hq, self.b_q]

        for param in self.parameters:
            param.attach_grad()

        is_random_iter = kwargs.get("is_random_iter", False)
        self._train(corpus_indices, num_steps, hyper_params, epochs, is_random_iter)

    def forward(self, inputs, state):
        """ forward function """
        h, = state
        outputs = []
        for x in inputs:
            z = nd.sigmoid(nd.dot(x, self.w_xz) + nd.dot(h, self.w_hz) + self.b_z)
            r = nd.sigmoid(nd.dot(x, self.w_xr) + nd.dot(h, self.w_hr) + self.b_r)
            h_tilda = nd.tanh(nd.dot(x, self.w_xh) + nd.dot(h, self.w_hh) + self.b_h)
            h = z * h + (1 - z) * h_tilda
            y = nd.dot(h, self.w_hq) + self.b_q
            outputs.append(y)
        y_hat = nd.concat(*outputs, dim=0)
        return y_hat, (h,)


if __name__ == '__main__':
    _corpus_indices, _idx_to_char, _char_to_idx, _vocab_size = \
        load_jaychou_lyrics("../data/jaychou_lyrics.txt.zip")
    _num_hidden = 256
    _num_steps = 35
    _batch_size = 32
    _lr = 1e2
    use_gluon = True
    if use_gluon:
        _rnn_layer = rnn.GRU(_num_hidden)
        model = BaseRNNGluon(_vocab_size, _idx_to_char, _char_to_idx, _num_hidden, _rnn_layer)
    else:
        model = GRUScratch(_vocab_size, _idx_to_char, _char_to_idx, _num_hidden)

    model.fit(_corpus_indices, _num_steps, {"lr": _lr, "batch_size": _batch_size}, epochs=250)
