#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: ch06_rnn.py
Date: 2019/8/21 2:51 PM
"""
import math
from mxnet import nd, autograd
from mxnet.gluon import nn, loss as gloss
from common import data_sets, log_utils, utils
from common.base_model import BaseClassifier


logger = log_utils.get_logger(__name__)


class RNNScratch(BaseClassifier):
    """
    """
    def __init__(self, **kwargs):
        super(RNNScratch, self).__init__(**kwargs)
        self.ctx = kwargs.get("context")
        self.num_inputs = kwargs["num_inputs"]
        self.num_hidden = kwargs["num_hidden"]
        self.num_outputs = kwargs["num_outputs"]
        self.vocab_size = kwargs["vocab_size"]
        self.idx_to_char = kwargs["idx_to_char"]
        self.char_to_idx = kwargs["char_to_idx"]
        self.w_xh = None
        self.w_hh = None
        self.b_h = None
        self.w_ho = None
        self.b_o = None
        self.batch_size = None

    def fit(self, train_iter, lr=0.03, batch_size=64, epochs=10, test_iter=None):
        """ fit data """
        def _get_params(shape):
            return nd.random.normal(scale=0.01, shape=shape, ctx=self.ctx)

        self.batch_size = batch_size
        self.w_xh = _get_params((self.num_inputs, self.num_hidden))
        self.w_hh = _get_params((self.num_hidden, self.num_hidden))
        self.b_h = nd.zeros(self.num_hidden, ctx=self.ctx)

        self.w_ho = _get_params((self.num_hidden, self.num_outputs))
        self.b_o = nd.zeros(self.num_outputs, ctx=self.ctx)

        self.params = [self.w_xh, self.w_hh, self.b_h, self.w_ho, self.b_o]
        # for param in self.params:
        #    param.attach_grad()

    def forward(self, inputs, state):
        """ forward function """
        h, = state
        outputs = []
        for x in inputs:
            h = nd.tanh(nd.dot(x, self.w_xh) + nd.dot(h, self.w_hh) + self.b_h)
            y = nd.dot(h, self.w_ho) + self.b_o
            outputs.append(y)
        return outputs, (h,)

    def init_state(self):
        """ init hidden state """
        return nd.zeros(shape=(self.batch_size, self.num_hidden), ctx=self.ctx),

    def predict_rnn(self, prefix, num_chars):
        """ predict sequence """
        state = self.init_state()
        output = [self.char_to_idx[prefix[0]]]
        for t in range(num_chars + len(prefix) - 1):
            x = data_sets.to_onehot(nd.array([output[-1]], ctx=self.ctx), self.vocab_size)
            y, state = self.forward(x, state)
            if t < len(prefix) - 1:
                output.append(self.char_to_idx[prefix[t + 1]])
            else:
                output.append(int(y[0].argmax(axis=1).asscalar()))
        return "".join([self.idx_to_char[i] for i in output])


if __name__ == '__main__':

    corpus_indices, char_to_idx, idx_to_char, vocab_size = \
        data_sets.load_jaychou_lyrics("../data/jaychou_lyrics.txt.zip")

    _num_inputs, _num_hidden, _num_outputs = vocab_size, 256, vocab_size
    context = utils.try_gpu()
    model = RNNScratch(ctx=context, num_inputs=_num_inputs, num_hidden=_num_hidden, num_outputs=_num_outputs,
                       vocab_size=vocab_size, char_to_idx=char_to_idx, idx_to_char=idx_to_char)
    model.fit(None, batch_size=1)
    init_state = model.init_state()
    # _outputs, _state = model.forward(_inputs, init_state)
    # logger.info("{}, {}, {}".format(len(_outputs), _outputs[0].shape, _state[0].shape))
    logger.info(model.predict_rnn("分开", 10))
