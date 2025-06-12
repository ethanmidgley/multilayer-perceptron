class SGD:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate

    def new_pass(self):
        pass

    def optimise_layer(self, layer):

        layer.weights -= layer.weight_dvector * self.learning_rate
        layer.bias -= layer.bias_dvector * self.learning_rate

    def post_optimise(self):
        pass
