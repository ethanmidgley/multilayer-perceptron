import numpy as np


class SGD:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate

    def new_pass(self, epoch):
        pass

    def optimise_layer(self, layer):

        # update weights & biases
        layer.weights -= layer.weight_dvector * self.learning_rate
        layer.bias -= layer.bias_dvector * self.learning_rate

        # clear weight & bias vectors
        layer.weight_dvector = None
        layer.bias_dvector = None

    def post_optimise(self):
        pass


class DecaySGD:
    def __init__(self, initial_rate=0.1, decay_after=5, decay_rate=0.1):

        # this will be the epoch at which after the learning rate will start to decay
        # the very first epoch is epoch 1
        self.decay_after = decay_after
        self.learning_rate = initial_rate
        self.decay_rate = decay_rate

    def new_pass(self, epoch):

        # if we are not at the threshold to decay yet return, do not alter learning rate
        if epoch < self.decay_after:
            return

        self.learning_rate *= np.exp(-self.decay_rate)

    def optimise_layer(self, layer):

        # update weights & biases
        layer.weights -= layer.weight_dvector * self.learning_rate
        layer.bias -= layer.bias_dvector * self.learning_rate

        # clear weight & bias vectors
        layer.weight_dvector = None
        layer.bias_dvector = None

    def post_optimise(self):
        pass
