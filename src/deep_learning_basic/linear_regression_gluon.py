#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from mxnet import autograd
from mxnet.gluon import data as gdata
from mxnet.gluon import nn
from mxnet.gluon import loss as gloss
from mxnet import gluon
from mxnet import init
sys.path.append("../")
from utils import data_utils
from utils import log_utils


logger = log_utils.get_logger(__name__)


net = nn.Sequential()
net.add(nn.Dense(1))
net.initialize(init.Normal(sigma=0.01))

loss = gloss.L2Loss()


def train(x, y, batch_size=10, epochs=3):
    global net
    global loss
    data_set = gdata.ArrayDataset(x, y)
    data_iter = gdata.DataLoader(data_set, batch_size, shuffle=True)
    trainer = gluon.Trainer(net.collect_params(), "sgd",
                            {"learning_rate": 0.03})
    for epoch in range(epochs):
        for epoch_x, epoch_y in data_iter:
            with autograd.record():
                epoch_loss = loss(net(epoch_x), epoch_y)
            epoch_loss.backward()
            trainer.step(batch_size)
        train_loss = loss(net(x), y)
        logger.info("epoch {}, loss: {}".format(epoch + 1,
                                                train_loss.mean().asnumpy()))


if __name__ == "__main__":

    batch_size = 64
    x, y = data_utils.load_data()
    train(x, y)

