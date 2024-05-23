from scipy import signal

import numpy as np

from src.layer.Layer import Layer
from src.layer.Sigmoid import Sigmoid


class ConvalutionalLayer(Layer):
    def __init__(self, input_dim, kernel_size, depth):
        input_depth, input_height, input_width = input_dim
        self.num_kernels = depth
        self.input_dim = input_dim
        self.input_depth = input_depth
        self.output_dim = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_dim = (depth, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_dim)
        self.biases = np.random.randn(*self.output_dim)
        self.sigmoid = Sigmoid()

    def forward_propagation(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.num_kernels):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")
        return self.sigmoid.activate(self.output)

    def backward_propagation(self, output_grad, learning_rate):
        output_grad = self.sigmoid.inverse_activate(output_grad)

        kernels_grad = np.zeros(self.kernels_dim)
        input_grad = np.zeros(self.input_dim)

        for i in range(self.num_kernels):
            for j in range(self.input_depth):
                kernels_grad[i, j] = signal.correlate2d(self.input[j], output_grad[i], "valid")
                input_grad[j] += signal.convolve2d(output_grad[i], self.kernels[i, j], "full")
        self.kernels -= learning_rate * kernels_grad
        self.biases -= learning_rate * output_grad
        return input_grad
