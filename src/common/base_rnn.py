#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: base_rnn.py
Date: 2019/8/22 8:24 PM
"""
import time
import math
from mxnet import nd, autograd, gluon, init
from mxnet.gluon import nn, loss as gloss
from functions import try_gpu, grad_clipping, sgd
from common.data_sets import data_iter_consecutive, data_iter_random
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
        self.ctx = try_gpu()
        self.init_scale = kwargs.get("init_scale", 0.01)

        self.trainer = None
        self.parameters = None
        self.loss = gloss.SoftmaxCrossEntropyLoss()

    def begin_state(self, batch_size):
        """ begin init state """
        raise NotImplementedError

    def fit(self, corpus_indices, num_steps, hyper_params, epochs, **kwargs):
        """fit function"""
        raise NotImplementedError

    def forward(self, inputs, state):
        """ forward function """
        raise NotImplementedError

    def _train(self, corpus_indices, num_steps, hyper_params, epochs, is_random_iter):
        """ train function """
        if is_random_iter:
            data_iter_fn = data_iter_random
        else:
            data_iter_fn = data_iter_consecutive

        batch_size = hyper_params.get("batch_size", 32)
        clipping_theta = hyper_params.get("clipping_theta", 1e-2)
        lr = hyper_params.get("lr", 1e2)

        history_loss = []
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
                    inputs = nd.one_hot(x.T, self.vocab_size)
                    y_hat, state = self.forward(inputs, state)
                    y = y.T.reshape((-1,))
                    batch_loss = self.loss(y_hat, y).mean()
                batch_loss.backward()

                if not self.parameters or len(self.parameters) <= 0:
                    self.parameters = [p.data() for p in self.collect_params().values()]
                grad_clipping(self.parameters, clipping_theta, self.ctx)

                if self.trainer:
                    self.trainer.step(1)
                else:
                    sgd(self.parameters, lr, 1)

                total_num += y.size
                total_loss += batch_loss.asscalar() * y.size

            history_loss.append(total_loss/total_num)
            if (epoch + 1) % 50 == 0:
                print("epoch {}, perplexity {}, time {} sec"
                      .format(epoch + 1, math.exp(total_loss / total_num), time.time() - start))
                print(self.predict_rnn("分开", 50))
                print(self.predict_rnn("不分开", 50))
        return history_loss

    def predict_rnn(self, prefix, num_chars):
        """ predict sequence """
        state = self.begin_state(1)
        output = [self.char_to_idx[prefix[0]]]

        for t in range(num_chars + len(prefix) - 1):
            inputs = nd.one_hot(nd.array([output[-1]], ctx=self.ctx).reshape((-1, 1)), self.vocab_size)
            y_hat, state = self.forward(inputs, state)
            if t < len(prefix) - 1:
                output.append(self.char_to_idx[prefix[t + 1]])
            else:
                output.append(int(y_hat[0].argmax(axis=0).asscalar()))
        return "".join(self.idx_to_char[i] for i in output)


class BaseRNNScratch(BaseRNN):
    """
    """
    def __init__(self, vocab_size, idx_to_char, char_to_idx, num_hidden, **kwargs):
        super(BaseRNNScratch, self).__init__(vocab_size, idx_to_char, char_to_idx, num_hidden, **kwargs)

    def begin_state(self, batch_size):
        raise NotImplementedError

    def fit(self, corpus_indices, num_steps, hyper_params, epochs, **kwargs):
        raise NotImplementedError

    def forward(self, inputs, state):
        raise NotImplementedError

    def _one_params(self, shape):
        return nd.random.normal(shape=shape, scale=self.init_scale, ctx=self.ctx)

    def _three_params(self):
        return (self._one_params(shape=(self.vocab_size, self.num_hidden)),
                self._one_params(shape=(self.num_hidden, self.num_hidden)),
                nd.zeros(self.num_hidden, ctx=self.ctx))


class BaseRNNGluon(BaseRNN):
    """
    """
    def __init__(self, vocab_size, idx_to_char, char_to_idx, num_hidden, rnn_layer, **kwargs):
        super(BaseRNNGluon, self).__init__(vocab_size, idx_to_char, char_to_idx, num_hidden, **kwargs)
        # model params
        self.rnn_layer = rnn_layer
        self.dense = nn.Dense(self.vocab_size)

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
    pass
