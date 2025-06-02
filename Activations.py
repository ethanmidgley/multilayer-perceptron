import numpy as np


class RELU:

    @staticmethod
    def activate(x):
        return np.maximum(0, x)  # return [max(0, z) for z in x]

    @staticmethod
    def deriv(x):
        return x > 0


class Linear:
    @staticmethod
    def activate(x):
        return x

    @staticmethod
    def deriv(x):
        return [1 for _ in range(len(x))]


class Sigmoid:
    @staticmethod
    def activate(x):
        result = 1 / (1 + np.exp(-x))
        return result

    @staticmethod
    def deriv(x):
        fx = Sigmoid.activate(x)
        return fx * (1 - fx)


class Softmax:
    @staticmethod
    def activate(x):

        shifted_x = x - np.max(x)
        exp_x = np.exp(shifted_x)
        sum_exp_x = np.sum(exp_x)

        return exp_x / sum_exp_x

    @staticmethod
    def deriv(x):
        fx = Softmax.activate(x)
        jacobian_matrix = np.diag(fx) - np.outer(fx, fx)  # Jacobian matrix
        return jacobian_matrix

        fx = Softmax.activate(x)
        return fx * (1 - fx)
