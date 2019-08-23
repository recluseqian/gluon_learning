#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: base_rnn.py
Date: 2019/8/22 8:24 PM
"""
import time
import mxnet as mx
from mxnet import nd, gluon, autograd
from mxnet.gluon import nn
from functions import try_gpu, grad_clipping, sgd
from common.data_sets import data_iter_consecutive, data_iter_random, to_onehot
import log_utils


logger = log_utils.get_logger(__name__)


class BaseRNN(nn.Block):
    """
    """

    def __init__(self, vocab_size, idx_to_char, char_to_idx, num_hidden, **kwargs):
        super(BaseRNN, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.idx_to_char = idx_to_char
        self.char_to_idx = char_to_idx
        self.num_hidden = num_hidden
        self.ctx = kwargs.get("ctx", try_gpu())
        self.init_scale = kwargs.get("init_scale", 0.01)

        self.trainer = None
        self.parameters = None
        self.loss = None

    def _train(self, corpus_indices, num_steps, hyper_params, epochs, is_random_iter):
        """ train function """
        if is_random_iter:
            data_iter_fn = data_iter_random
        else:
            data_iter_fn = data_iter_consecutive

        batch_size = hyper_params.get("batch_size", 32)
        clipping_theta = hyper_params.get("clipping_theta", 1e-2)
        lr = hyper_params.get("lr", 1e2)

        for epoch in range(epochs):
            total_loss, total_num, start = 0.0, 0, time.time()

            state = None
            if not is_random_iter:
                state = self.begin_state(batch_size)

            data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, ctx=self.ctx)
            for x, y in data_iter:
                if is_random_iter:
                    state = self.begin_state(batch_size)
                else:
                    for s in state:
                        s.detach()

                with autograd.record():
                    inputs = to_onehot(x, self.vocab_size)
                    y_hat, state = self.forward(inputs, state)
                    y = y.T.reshape((-1,))
                    batch_loss = self.loss(y_hat, y).mean()
                batch_loss.backward()

                grad_clipping(self.parameters, clipping_theta, self.ctx)
                if self.trainer:
                    self.trainer.step(1)
                else:
                    sgd(self.parameters, lr, 1)

                total_num += y.size
                total_loss += batch_loss.asscalar() * y.size

    def begin_state(self, *args, **kwargs):
        """ begin init state """
        raise NotImplementedError

    def forward(self, *args):
        """ forward function """
        raise NotImplementedError

    def predict_rnn(self, prefix, num_chars):
        """ predict sequence """
        state = self.begin_state(1)
        output = [self.char_to_idx[prefix[0]]]

        for t in range(num_chars + len(prefix) - 1):
            inputs = to_onehot(nd.array([output[-1]], ctx=self.ctx).reshape((1, -1)), self.vocab_size)
            y_hat, state = self.forward(inputs, state)
            output.append(y_hat.argmax(axix=0).asscalar())
        return "".join(self.idx_to_char[i] for i in output)


if __name__ == '__main__':
    pass
