import numpy as np

from src.layer.Layer import Layer


class Network:
    def __init__(self, layers: [Layer]):
        self.layers = layers

    def fit(self, x_train, y_train, epochs, learning_rate):
        for epoch in range(epochs):
            for x, y in zip(x_train, y_train):
                output = self.predict(x)
                grad = self.__binary_cross_entropy_prime(y, output)
                for layer in reversed(self.layers):
                    grad = layer.backward_propagation(grad, learning_rate)

    def predict(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward_propagation(output)
        return output

    def predictions(self, inputs):
        predicted = []
        for input in inputs:
            output = self.predict(input)
            predicted.append(output)
        return predicted


    def __binary_cross_entropy_prime(self, y_true, y_pred):
        return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)
