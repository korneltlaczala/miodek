import matplotlib.pyplot as plt
import numpy as np
from activation_functions import *

def plot(X, func):
    Y = func.activate(X)
    plt.scatter(X, Y, label=f"{func.name} Activation", color='b', s=10, alpha=0.8)
    plt.xlabel("Input X")
    plt.ylabel("Output Y")
    plt.title(f"Scatter Plot of {func.name} Activation Function")
    plt.legend()
    plt.grid(True)
    plt.show()




X = np.random.uniform(-5, 5, size=(10000,))
funcs = [ReLU(), Sigmoid(), LeakyReLU(), ELU()]
for func in funcs:
    plot(X, func)

