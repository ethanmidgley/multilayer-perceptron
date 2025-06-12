import numpy as np


class Dense:
    def __init__(self, n_perceptrons, activation):
        self.prev_layer = None
        self.next_layer = None
        self.layer_number = None

        self.weights = np.array([])
        self.bias = np.array([])

        self.n_perceptrons = n_perceptrons
        self.activation = activation

        self.z = []
        self.a = []

        self.inputs = np.array([])

        # gradient vectors
        self.weight_dvector = None
        self.bias_dvector = None

        self.is_trainable = True

    def feed(self, inputs, training):

        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.bias
        self.a = self.activation.activate(self.z)

        if self.next_layer is not None:
            return self.next_layer.feed(self.a, training)

        return self.a

    def shape(self):
        return self.n_perceptrons

    def compile(self, layer_number, previous_layer, next_layer):

        self.layer_number = layer_number
        self.next_layer = next_layer
        self.prev_layer = previous_layer

        self.weights = 0.01 * np.random.randn(
            self.prev_layer.shape(), self.n_perceptrons
        )

        self.bias = np.zeros((1, self.n_perceptrons))

    def backprop(self, val):

        if self.prev_layer is None:
            return

        pc_pz = self.activation.backprop(self.z, val)

        # weight gradients
        pc_pw = np.dot(self.inputs.T, pc_pz)

        # bias gradients
        pc_pb = np.sum(pc_pz, axis=0, keepdims=True)

        # we need the cost over the inputs to give to the previous layer to back prop
        # only if we are not the last trainable layer
        pc_pi = np.dot(pc_pz, self.weights.T)

        # store the gradient vectors for the optimiser to use
        self.weight_dvector = pc_pw
        self.bias_dvector = pc_pb

        self.prev_layer.backprop(pc_pi)
