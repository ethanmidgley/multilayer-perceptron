import numpy as np


class Dropout:

    def __init__(self, rate):
        self.rate = 1 - rate

        self.n_perceptrons = 0

        self.prev_layer = None
        self.next_layer = None
        self.layer_number = None

        self.mask = None
        self.inputs = None
        self.z = None

        self.is_trainable = False

    def feed(self, inputs, training):

        if not training:
            if self.next_layer is not None:
                return self.next_layer.feed(inputs, training)
            return inputs

        self.inputs = inputs
        self.mask = np.random.binomial(1, self.rate, inputs.shape) / self.rate

        self.z = self.mask * inputs

        if self.next_layer is not None:
            return self.next_layer.feed(self.z, training)
        return self.z

    def shape(self):
        return self.n_perceptrons

    def compile(self, layer_number, previous_layer, next_layer):

        self.layer_number = layer_number
        self.next_layer = next_layer
        self.prev_layer = previous_layer
        self.n_perceptrons = previous_layer.shape()

    def backprop(self, val):

        if self.prev_layer is None:
            return

        pc_pm = val * self.mask

        self.prev_layer.backprop(pc_pm)
