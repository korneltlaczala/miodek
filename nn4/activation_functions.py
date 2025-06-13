from abc import abstractmethod
from matplotlib import pyplot as plt
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
        X = X.clip(-100, 100)
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
        return np.ones_like(X)

class Softmax(ActivationFunction):
    def __init__(self):
        self.name = "Softmax"

    def activate(self, X):
        e_x = np.exp(X - np.max(X, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def derivative(self, X):
        s = self.activate(X)
        return s * (1 - s)  # This is a simplified version; true Jacobian is more complex.

def test(function):
    x = np.linspace(-5, 5, 100)
    plt.figure(figsize=(10, 6))
    plt.plot(x, function.activate(x), label="Activation", color="blue", linestyle="-")
    plt.plot(x, function.derivative(x), label="Derivative", color="orange", linestyle="--")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title(f"{function.name} Activation Function")
    plt.show()


if __name__ == "__main__":
    test(ReLU())

