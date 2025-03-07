import matplotlib.pyplot as plt
import numpy as np
from activation_functions import *

def plot_activation(X, func):
    y = func.activate(X)
    plot(X, y, type="activation")

def plot_derivative(X, func):
    y = func.derivative(X)
    plot(X, y, type="derivative")

def plot(X, y, type):
    plt.scatter(X, y, label=f"{func.name} {type.capitalize()}", color='b', s=10, alpha=0.8)
    plt.xlabel("Input X")
    plt.ylabel("Output y")
    plt.title(f"Scatter Plot of {func.name} {type.capitalize()} Function")
    plt.legend()
    plt.grid(True)
    plt.show()




X = np.random.uniform(-5, 5, size=(10000,))
funcs = [ReLU(), Sigmoid(), LeakyReLU(), ELU()]
for func in funcs:
    plot_activation(X, func)
    plot_derivative(X, func)
