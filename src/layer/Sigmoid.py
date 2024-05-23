import numpy as np


class Sigmoid:
    def __init__(self):
        self.input = None

    def activate(self, input):
        self.input = input
        res = 1 / (1 + np.exp(-input))
        return res

    def inverse_activate(self, output_gradient):
        return np.multiply(output_gradient, self.activation_prime(self.input))

    def activation_prime(self, input):
        s = self.activate(input)
        res = s * (1 - s)
        return res
