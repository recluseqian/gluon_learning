#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: ch06_lstm.py
Date: 2019/8/24 6:04 PM
"""
from mxnet import nd
from mxnet.gluon import rnn
from common.base_rnn import BaseRNNScratch, BaseRNNGluon
from common.data_sets import load_jaychou_lyrics


class LSTMScratch(BaseRNNScratch):
    """
    """

    def __init__(self, vocab_size, idx_to_char, char_to_idx, num_hidden, **kwargs):
        super(LSTMScratch, self).__init__(vocab_size, idx_to_char, char_to_idx, num_hidden, **kwargs)
        self.w_xi, self.w_hi, self.b_i = None, None, None
        self.w_xf, self.w_hf, self.b_f = None, None, None
        self.w_xo, self.w_ho, self.b_o = None, None, None
        self.w_xc, self.w_hc, self.b_c = None, None, None
        self.w_hq, self.b_q = None, None

    def begin_state(self, batch_size):
        """init first state"""
        return (nd.zeros(shape=(batch_size, self.num_hidden), ctx=self.ctx),
                nd.zeros(shape=(batch_size, self.num_hidden), ctx=self.ctx))

    def fit(self, corpus_indices, num_steps, hyper_params, epochs, **kwargs):
        """ fit function """
        self.w_xi, self.w_hi, self.b_i = self._three_params()
        self.w_xf, self.w_hf, self.b_f = self._three_params()
        self.w_xo, self.w_ho, self.b_o = self._three_params()
        self.w_xc, self.w_hc, self.b_c = self._three_params()
        self.w_hq = self._one_params(shape=(self.num_hidden, self.vocab_size))
        self.b_q = nd.zeros(shape=(self.vocab_size,), ctx=self.ctx)

        self.parameters = [
            self.w_xi, self.w_hi, self.b_i,
            self.w_xf, self.w_hf, self.b_f,
            self.w_xo, self.w_ho, self.b_o,
            self.w_xc, self.w_hc, self.b_c,
            self.w_hq, self.b_q
        ]

        for param in self.parameters:
            param.attach_grad()

        is_random_iter = kwargs.get("is_random_iter", False)
        self._train(corpus_indices, num_steps, hyper_params, epochs, is_random_iter)

    def forward(self, inputs, state):
        """ forward function """
        h, c = state
        outputs = []
        for x in inputs:
            i = nd.sigmoid(nd.dot(x, self.w_xi) + nd.dot(h, self.w_hi) + self.b_i)
            f = nd.sigmoid(nd.dot(x, self.w_xf) + nd.dot(h, self.w_hf) + self.b_f)
            o = nd.sigmoid(nd.dot(x, self.w_xo) + nd.dot(h, self.w_ho) + self.b_o)

            c_tilda = nd.tanh(nd.dot(x, self.w_xc) + nd.dot(h, self.w_hc) + self.b_c)
            c = f * c + i * c_tilda
            h = o * c

            y = nd.dot(h, self.w_hq) + self.b_q
            outputs.append(y)

        y_hat = nd.concat(*outputs, dim=0)
        return y_hat, (h, c)


if __name__ == '__main__':
    _corpus_indices, _idx_to_char, _char_to_idx, _vocab_size = \
        load_jaychou_lyrics("../data/jaychou_lyrics.txt.zip")
    _num_hidden = 256
    _num_steps = 35
    _batch_size = 32
    _lr = 1e2
    use_gluon = False
    if use_gluon:
        _rnn_layer = rnn.LSTM(_num_hidden)
        model = BaseRNNGluon(_vocab_size, _idx_to_char, _char_to_idx, _num_hidden, _rnn_layer)
    else:
        model = LSTMScratch(_vocab_size, _idx_to_char, _char_to_idx, _num_hidden)

    model.fit(_corpus_indices, _num_steps, {"lr": _lr, "batch_size": _batch_size}, epochs=250)
