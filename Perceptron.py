import numpy as np


class Perceptron:

    def __init__(self, n_weights, activation):
        self.activation = activation
        self.weights = [(1 / n_weights) for _ in range(n_weights)]

    def predict(self, inputs, bias):
        return np.dot(self.weights, inputs) + bias
