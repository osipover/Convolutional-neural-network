import numpy as np

from src.layer.Layer import Layer


class ReshapeLayer(Layer):
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward_propagation(self, input):
        return np.reshape(input, self.output_dim)

    def backward_propagation(self, output_grad, learning_rate):
        return np.reshape(output_grad, self.input_dim)