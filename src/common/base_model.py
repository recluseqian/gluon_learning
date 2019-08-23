#!/usr/bin/env python
# -*- coding:utf8 -*-
import time
import math
from mxnet import nd, autograd
from common.functions import sgd, grad_clipping
from . import log_utils, data_sets


logger = log_utils.get_logger(__name__)


class BaseRegression:
    """
    """
    def __init__(self, **kwargs):
        self.params = []
        self.trainers = []
        self.loss = kwargs.get('loss')

    def fit(self, train_x, train_y, lr, batch_size, epochs, test_x, test_y):
        """
        """
        raise NotImplementedError

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

    def _train(self, x, y, lr, batch_size):
        """
        """
        with autograd.record():
            y_hat = self.forward(x)
            batch_loss = self._loss(y_hat, y)
        batch_loss.backward()
        if self.trainers:
            [trainer.step(batch_size) for trainer in self.trainers]
        else:
            sgd(self.params, lr, batch_size)

    def _loss(self, y_hat, y):
        """ loss function """
        return self.loss(y_hat, y)


class BaseClassifier:
    """
    """
    def __init__(self, **kwargs):
        self.params = []
        self.trainers = []
        self.loss = None

    def fit(self, train_iter, lr, batch_size, epochs, test_iter):
        """
        """
        raise NotImplementedError

    def forward(self, *args):
        raise NotImplementedError

    def evaluate_accuracy(self, data_iter):
        """
        """
        acc_num, total_num = 0.0, 0.0
        for x, y in data_iter:
            y = y.astype('float32')
            y_hat = self.forward(x)
            acc_num += (y_hat.argmax(axis=1) == y).sum().asscalar()
            total_num += y.size
        return acc_num / total_num

    def _train(self, train_iter, lr, batch_size, epochs, test_iter=None):
        """
        """

        for epoch in range(epochs):
            train_loss, train_total, train_acc = 0.0, 0, 0.0
            for x, y in train_iter:
                with autograd.record():
                    y_hat = self.forward(x)
                    print(y_hat.shape)
                    print(y.shape)
                    batch_loss = self._loss(y_hat, y).sum()
                batch_loss.backward()

                if self.trainers:
                    [trainer.step(batch_size) for trainer in self.trainers]
                else:
                    sgd(self.params, lr, batch_size)

                train_loss += batch_loss.asscalar()
                y = y.astype("float32")
                train_acc += (y_hat.argmax(axis=1) == y).sum().asscalar()
                train_total += y.size

            train_loss /= train_total
            train_acc /= train_total

            if test_iter:
                test_acc = self.evaluate_accuracy(test_iter)
                logger.info("epoch %d, loss: %.4f, train acc %.4f, test acc: %.4f"
                            % (epoch + 1, train_loss, train_acc, test_acc))

    def _loss(self, y_hat, y):
        """ loss function """
        return self.loss(y_hat, y)


class BaseRNN:
    """
    """
    def __init__(self, vocab_size, idx_to_char, char_to_idx, num_hidden, ctx, **kwargs):
        self.vocab_size = vocab_size
        self.idx_to_char = idx_to_char
        self.char_to_idx = char_to_idx
        self.num_hidden = num_hidden
        self.ctx = ctx
        self.params = None
        self.loss = None
        self.init_scale = kwargs.get("init_scale", 0.01)
        logger.info("will use {}".format(self.ctx))

    def init_rnn_state(self, batch_size):
        """ init rnn state """
        return nd.zeros(shape=(batch_size, self.num_hidden), ctx=self.ctx),

    def fit(self, corpus_indices, lr, batch_size, epochs, num_steps, **kwargs):
        """ fit function """
        is_random_iter = kwargs.get("is_random_iter", False)
        clipping_theta = kwargs.get("clipping_theta", 1e-2)
        if is_random_iter:
            data_iter_fn = data_sets.data_iter_random
        else:
            data_iter_fn = data_sets.data_iter_consecutive

        for epoch in range(epochs):
            state = None
            if not is_random_iter:
                state = self.init_rnn_state(batch_size)

            total_loss, total_num, start = 0.0, 0, time.time()

            data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, ctx=self.ctx)
            for x, y in data_iter:
                if is_random_iter:
                    state = self.init_rnn_state(batch_size)
                else:
                    for s in state:
                        s.detach()

                with autograd.record():
                    inputs = data_sets.to_onehot(x, self.vocab_size)
                    outputs, state = self.forward(inputs, state)
                    outputs = nd.concat(*outputs, dim=0)
                    y = y.T.reshape((-1,))
                    batch_loss = self.loss(outputs, y).mean()
                batch_loss.backward()
                grad_clipping(self.params, clipping_theta, self.ctx)
                sgd(self.params, lr, 1)
                total_num += y.size
                total_loss += batch_loss.asscalar() * y.size
            if (epoch + 1) % 20 == 0:
                logger.info("epoch {}, perplexity {}, time {} sec"
                            .format(epoch + 1, math.exp(total_loss / total_num), time.time() - start))

                for prefix in ("分开", "不分开"):
                    logger.info(self.predict_rnn(prefix, 50))

    def forward(self, inputs, state):
        """ forward function """
        raise NotImplementedError

    def predict_rnn(self, prefix, num_chars):
        """ predict function based on prefix """
        state = self.init_rnn_state(1)
        output = [self.char_to_idx[prefix[0]]]

        for t in range(num_chars + len(prefix) - 1):
            x = data_sets.to_onehot(nd.array([output[-1]], ctx=self.ctx), self.vocab_size)
            y, state = self.forward(x, state)
            if t < len(prefix) - 1:
                output.append(self.char_to_idx[prefix[t + 1]])
            else:
                output.append(int(y[0].argmax(axis=1).asscalar()))
        return "".join([self.idx_to_char[i] for i in output])

    def _one_param(self, shape):
        return nd.random.normal(scale=self.init_scale, shape=shape, ctx=self.ctx)


if __name__ == '__main__':
    pass
