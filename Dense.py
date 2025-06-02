import numpy as np

from Perceptron import Perceptron


class Dense:
    def __init__(self, n_perceptrons, activation):
        self.prev_layer = None
        self.next_layer = None
        self.layer_number = None
        self.learning_rate = 0.1

        self.weights = np.array([])

        self.n_perceptrons = n_perceptrons
        self.activation = activation
        self.bias = np.array([])

        self.z = []
        self.a = []

        self.inputs = np.array([])

    def feed(self, inputs):

        # self.z = [node.predict(inputs, self.bias) for node in self.nodes]
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.bias
        self.a = self.activation.activate(self.z)

        if self.next_layer is not None:
            return self.next_layer.feed(self.a)

        return self.a

    def shape(self):
        return self.n_perceptrons

    def compile(self, layer_number, previous_layer, next_layer, learning_rate=0.1):

        self.layer_number = layer_number
        self.next_layer = next_layer
        self.prev_layer = previous_layer
        self.learning_rate = learning_rate

        # this the one I wrote, the uncommented is the one in the book
        # self.weights = np.random.rand(self.n_perceptrons, self.prev_layer.shape())
        # self.weights = 0.01 * np.random.randn(
        #     self.n_perceptrons, self.prev_layer.shape()
        # )
        self.weights = 0.01 * np.random.randn(
            self.prev_layer.shape(), self.n_perceptrons
        )

        # my bias, uncommented is the books
        # self.bias = np.zeros(1, self.n_perceptrons)
        self.bias = np.zeros((1, self.n_perceptrons))

    def backprop(self, val):
        """
        Will take in the calculated gradient of A[L]
        """

        if self.prev_layer is None:
            return

        # print("Backprop @ layer", self.layer_number, "Chained value", val)

        pa_pz = self.activation.deriv(self.z)

        pc_pz = pa_pz * val

        pc_pw = np.dot(self.inputs.T, pc_pz)

        # pc_pw = pc_pz * prev_activation

        # pc_pw can be gradient descented
        # pc_pb can be gradient descented

        # bias gradients
        # pc_pb = 1 * pc_pz
        pc_pb = np.sum(pc_pz, axis=0, keepdims=True)

        # we need the cost over the inputs to give to the previous layer to back prop
        # only if we are not the last trainable layer
        pc_pi = np.dot(pc_pz, self.weights.T)

        self.gradient_descent(pc_pw, pc_pb)
        # pz_over_prevlayer_activation = self.weights.T
        #
        self.prev_layer.backprop(pc_pi)

    def gradient_descent(self, dW, dB):
        # print("Descent running @ layer", self.layer_number)
        # print("Weights deriv", dW)
        # print("Bias deriv", dW)
        # print("Weights before", self.weights)
        self.weights -= self.learning_rate * dW
        self.bias -= self.learning_rate * dB
