import numpy as np


class RELU:

    @staticmethod
    def activate(x):
        return np.maximum(0, x)  # return [max(0, z) for z in x]

    @staticmethod
    def deriv(x):
        return x > 0

    @staticmethod
    def backprop(z_value, gradient_vectors):
        return RELU.deriv(z_value) * gradient_vectors


class Linear:
    @staticmethod
    def activate(x):
        return x

    @staticmethod
    def deriv(x):
        return [1 for _ in range(len(x))]

    @staticmethod
    def backprop(z_value, gradient_vectors):
        return Linear.deriv(z_value) * gradient_vectors


class Sigmoid:
    @staticmethod
    def activate(x):
        result = 1 / (1 + np.exp(-x))
        return result

    @staticmethod
    def deriv(x):
        fx = Sigmoid.activate(x)
        return fx * (1 - fx)

    @staticmethod
    def backprop(z_value, gradient_vectors):
        return Sigmoid.deriv(z_value) * gradient_vectors


class Softmax:
    @staticmethod
    def activate(x):

        shifted_x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted_x)
        sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)

        return exp_x / sum_exp_x

    @staticmethod
    def deriv(inputs):
        pass

    @staticmethod
    def backprop(z_value, gradient_vectors):

        output = Softmax.activate(z_value)

        grads = np.empty_like(gradient_vectors)

        for index, (single_output, single_d) in enumerate(
            zip(output, gradient_vectors)
        ):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(
                single_output, single_output.T
            )
            grads[index] = np.dot(jacobian_matrix, single_d)

        return grads
