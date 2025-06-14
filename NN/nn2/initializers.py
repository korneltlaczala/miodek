import numpy as np

class Initializer():
    def initialize(self, model):
        self.initialize_weights(model)
        self.initialize_biases(model)

    def initialize_weights(self, model):
        raise NotImplementedError

    def initialize_biases(self, model):
        raise NotImplementedError


class NormalInitializer(Initializer):
    def initialize_weights(self, model):
        for layer in model.layers:
            layer.weights = np.random.normal(size=(layer.neurons_in, layer.neurons_out))

    def initialize_biases(self, model):
        for layer in model.layers:
            layer.biases = np.random.normal(size=(layer.neurons_out))

class UniformInitializer(Initializer):
    def initialize_weights(self, model):
        for layer in model.layers:
            layer.weights = np.random.uniform(size=(layer.neurons_in, layer.neurons_out))

    def initialize_biases(self, model):
        for layer in model.layers:
            layer.biases = np.random.uniform(size=(layer.neurons_out))

class XavierNormalInitializer(Initializer):
    def initialize_weights(self, model):
        for layer in model.layers:
            sd = np.sqrt(2 / (layer.neurons_in + layer.neurons_out))
            layer.weights = np.random.normal(0, sd, size=(layer.neurons_in, layer.neurons_out))

    def initialize_biases(self, model):
        for layer in model.layers:
            layer.biases = np.random.normal(size=(layer.neurons_out))


class XavierUniformInitializer(Initializer):
    def initialize_weights(self, model):
        for layer in model.layers:
            bound = np.sqrt(6 / (layer.neurons_in + layer.neurons_out))
            layer.weights = np.random.uniform(-bound, bound, size=(layer.neurons_in, layer.neurons_out))

    def initialize_biases(self, model):
        for layer in model.layers:
            layer.biases = np.random.uniform(size=(layer.neurons_out))


class HeNormalInitializer(Initializer):
    def initialize_weights(self, model):
        for layer in model.layers:
            sd = np.sqrt(2 / layer.neurons_in)
            layer.weights = np.random.normal(0, sd, size=(layer.neurons_in, layer.neurons_out))

    def initialize_biases(self, model):
        for layer in model.layers:
            layer.biases = np.random.normal(size=(layer.neurons_out))


class HeUniformInitializer(Initializer):
    def initialize_weights(self, model):
        for layer in model.layers:
            bound = np.sqrt(6 / layer.neurons_in)
            layer.weights = np.random.uniform(-bound, bound, size=(layer.neurons_in, layer.neurons_out))

    def initialize_biases(self, model):
        for layer in model.layers:
            layer.biases = np.random.uniform(size=(layer.neurons_out))