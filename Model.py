import math

import numpy as np

from Metrics import MSE


class Model:

    def __init__(self):
        self.is_compiled = False
        self.metric = MSE
        self.learning_rate = 0
        self.layers = []

    def add(self, layer):
        """Adds a layer to the model"""

        assert not self.is_compiled
        self.layers.append(layer)

    def compile(self, metric, learning_rate=0.1):
        assert len(self.layers) > 1

        self.is_compiled = True
        self.learning_rate = learning_rate
        self.metric = metric

        for i, layer in enumerate(self.layers):
            layer.compile(
                i,
                self.layers[i - 1] if i != 0 else None,
                self.layers[i + 1] if i != len(self.layers) - 1 else None,
            )

    def feed_batch(self, batch):
        assert self.is_compiled
        return self.layers[0].feed(batch)

    def predict(self, row):
        return self.layers[0].feed(row)

    def train(self, rows, labels, epochs=5, batch_size=32):
        assert self.is_compiled

        train_steps = math.ceil(len(rows) / batch_size)

        for _ in range(epochs):

            for i in range(train_steps):

                batch_x = rows[i * batch_size : (i + 1) * batch_size]
                batch_y = labels[i * batch_size : (i + 1) * batch_size]

                # lets just say user manually feeds in a batch for the sake of xor

                yhat = self.feed_batch(batch_x)
                c = self.metric.calculate(batch_y, yhat)

                cost_gradient = self.metric.deriv(batch_y, yhat)

                self.layers[-1].backprop(cost_gradient)
