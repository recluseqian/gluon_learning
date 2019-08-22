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
from common.base_model import BaseRNN


logger = log_utils.get_logger(__name__)


class RNNScratch(BaseRNN):
    """
    """
    def __init__(self, vocab_size, idx_to_char, char_to_idx, num_hidden, ctx, **kwargs):
        super(RNNScratch, self).__init__(vocab_size, idx_to_char, char_to_idx, num_hidden, ctx, **kwargs)
        # model params
        self.w_xh = self._one_param(shape=(vocab_size, num_hidden))
        self.w_hh = self._one_param(shape=(num_hidden, num_hidden))
        self.b_h = nd.zeros(num_hidden, ctx=ctx)
        self.w_hq = self._one_param(shape=(num_hidden, vocab_size))
        self.b_q = nd.zeros(vocab_size, ctx=ctx)
        self.params = [self.w_xh, self.w_hh, self.b_h, self.w_hq, self.b_q]
        for param in self.params:
            param.attach_grad()

        self.loss = gloss.SoftmaxCrossEntropyLoss()

    def forward(self, inputs, state):
        """ forward function """
        h, = state
        outputs = []
        for x in inputs:
            h = nd.tanh(nd.dot(x, self.w_xh) + self.b_h)
            y = nd.dot(h, self.w_hq) + self.b_q
            outputs.append(y)
        return outputs, (h,)


def test_forward():
    """ test """
    _corpus_indices, _idx_to_char, _char_to_idx, _vocab_size = \
        data_sets.load_jaychou_lyrics("../data/jaychou_lyrics.txt.zip")

    context = utils.try_gpu()
    _num_hidden = 256
    _x = nd.arange(10).reshape((2, 5))
    _inputs = data_sets.to_onehot(_x.as_in_context(context), _vocab_size)
    model = RNNScratch(_vocab_size, _idx_to_char, _char_to_idx, _num_hidden, context)
    _state = model.init_rnn_state(2)
    _outputs, _state_new = model.forward(_inputs, _state)
    logger.info("{}, {}, {}".format(len(_outputs), _outputs[0].shape, _state_new[0].shape))


def test_predict():
    _corpus_indices, _idx_to_char, _char_to_idx, _vocab_size = \
        data_sets.load_jaychou_lyrics("../data/jaychou_lyrics.txt.zip")

    context = utils.try_gpu()
    _num_hidden = 256
    model = RNNScratch(_vocab_size, _idx_to_char, _char_to_idx, _num_hidden, context)
    result = model.predict_rnn("分开", 10)
    logger.info(result)


def test_train(args):
    _corpus_indices, _idx_to_char, _char_to_idx, _vocab_size = \
        data_sets.load_jaychou_lyrics(args.file_name)

    is_random_iter = args.is_random_iter == "1"

    context = utils.try_gpu()
    _num_hidden = 256
    model = RNNScratch(_vocab_size, _idx_to_char, _char_to_idx, _num_hidden, context, is_random_iter=is_random_iter)
    model.fit(_corpus_indices, lr=1e2, batch_size=32, epochs=250, num_steps=35)


if __name__ == '__main__':
    # test_forward()
    # test_predict()
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--file_name", default="../data/jaychou_lyrics.txt.zip")
    parser.add_argument("--is_random_iter", default="1")
    args = parser.parse_args()
    test_train(args)

