import numpy as np


class Input:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.prev_layer = None
        self.next_layer = None
        self.layer_number = None

        self.a = []

    def shape(self):
        return self.input_shape

    def compile(self, layer_number, previous_layer, next_layer):
        self.layer_number = layer_number
        self.next_layer = next_layer
        self.prev_layer = previous_layer

    def feed(self, inputs):

        # self.a = np.dot(np.array(inputs), 1)
        self.a = np.array(inputs)

        if self.next_layer is not None:
            return self.next_layer.feed(self.a)

        return self.a

    def backprop(self, _):
        return
