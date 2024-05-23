import abc


class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    @abc.abstractmethod
    def forward_propagation(self, input):
        # return output
        pass

    @abc.abstractmethod
    def backward_propagation(self, output_grad, learning_rate):
        # update kernels, biases and return input gradient
        pass
