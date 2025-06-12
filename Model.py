import math

from Metrics import MSE
from Optimisers import SGD


class Model:

    def __init__(self):
        self.is_compiled = False
        self.metric = MSE
        self.optimiser = SGD()
        self.layers = []
        self.trainable_layers = []

    def add(self, layer):
        """Adds a layer to the model"""

        assert not self.is_compiled

        if layer.is_trainable:
            self.trainable_layers.append(layer)

        self.layers.append(layer)

    def compile(self, metric, optimiser=SGD()):

        assert len(self.layers) > 1

        self.is_compiled = True
        self.optimiser = optimiser
        self.metric = metric

        for i, layer in enumerate(self.layers):
            prev_layer = self.layers[i - 1] if i != 0 else None
            next_layer = self.layers[i + 1] if i != len(self.layers) - 1 else None
            layer.compile(
                i,
                prev_layer,
                next_layer,
            )

    def feed_batch(self, batch, training=False):
        assert self.is_compiled
        return self.layers[0].feed(batch, training)

    def predict(self, row):
        return self.layers[0].feed(row, False)

    def train(self, rows, labels, epochs=5, batch_size=32):
        assert self.is_compiled

        train_steps = math.ceil(len(rows) / batch_size)

        for _ in range(epochs):

            self.optimiser.new_pass()

            for i in range(train_steps):

                batch_x = rows[i * batch_size : (i + 1) * batch_size]
                batch_y = labels[i * batch_size : (i + 1) * batch_size]

                # lets just say user manually feeds in a batch for the sake of xor

                yhat = self.feed_batch(batch_x, training=True)
                _cost = self.metric.calculate(batch_y, yhat)

                # Calculate cost derivative and start backwards propagation
                cost_gradient = self.metric.deriv(batch_y, yhat)
                self.layers[-1].backprop(cost_gradient)

                # now we can run our optimiser on the trainable layers
                for trainable in self.trainable_layers:
                    self.optimiser.optimise_layer(trainable)
                self.optimiser.post_optimise()
