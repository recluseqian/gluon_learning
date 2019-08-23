#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: ch06_rnn_v2.py
Date: 2019/8/23 7:11 PM
"""
from mxnet import nd
from common.base_rnn import BaseRNN


class RNNScratch(BaseRNN):
    """
    """
    def __init__(self, vocab_size, idx_to_char, char_to_idx, num_hidden, **kwargs):
        super(RNNScratch, self).__init__(vocab_size, idx_to_char, char_to_idx, num_hidden, **kwargs)
        self.w_xh = None

    def begin_state(self, *args, **kwargs):
        pass

    def forward(self, *args):
        pass

    def _one_params(self, shape):
        return nd.random.normal(scale=self.init_scale, shape=shape, ctx=self.ctx)

