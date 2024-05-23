import numpy as np

from src.layer.Layer import Layer
from src.layer.Sigmoid import Sigmoid


class DenseLayer(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)
        self.sigmoid = Sigmoid()

    def forward_propagation(self, input):
        self.input = input
        result = np.dot(self.weights, self.input) + self.bias
        return self.sigmoid.activate(result)

    def backward_propagation(self, output_grad, learning_rate):
        output_grad = self.sigmoid.inverse_activate(output_grad)
        weights_grad = np.dot(output_grad, self.input.T)
        input_grad = np.dot(self.weights.T, output_grad)
        self.weights -= learning_rate * weights_grad
        self.bias -= learning_rate * output_grad
        return input_grad
