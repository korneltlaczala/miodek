from abc import abstractmethod
import numpy as np


class ActivationFunction():

    @abstractmethod
    def activate(self, X):
        pass

    @abstractmethod
    def derivative(self, X):
        pass

    def get_name(self):
        return self.name

    def __str__(self):
        return self.name

class Sigmoid(ActivationFunction):

    def __init__(self):
        self.name = "Sigmoid"

    def activate(self, X):
        X.clip(-500, 500)
        return 1 / (1 + np.exp(-X))

    def derivative(self, X):
        y = self.activate(X)
        return y * (1 - y)

class ReLU(ActivationFunction):

    def __init__(self):
        self.name = "ReLU"

    def activate(self, X):
        return np.maximum(0, X)

    def derivative(self, X):
        return np.where(X > 0, 1, 0)

class LeakyReLU(ActivationFunction):

    def __init__(self, alpha=0.1):
        self.name = "LeakyReLU"
        self.alpha = alpha

    def activate(self, X):
        return np.maximum(self.alpha * X, X)

    def derivative(self, X):
        return np.where(X > 0, 1, self.alpha)

class ELU(ActivationFunction):

    def __init__(self):
        self.name = "ELU"

    def activate(self, X):
        return np.where(X > 0, X, np.exp(X) - 1)

    def derivative(self, X):
        return np.where(X > 0, 1, np.exp(X))


class Linear(ActivationFunction):

    def __init__(self):
        self.name = "Linear"

    def activate(self, X):
        return X

    def derivative(self, X):
        return 1