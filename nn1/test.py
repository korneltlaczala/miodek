import numpy as np
import pandas as pd


# A = np.array([[1, 1], [2, 2], [3, 3], [4,4], [5,5]])
# X = np.array([[0,0,0], [1,1,1]])

# df = pd.read_csv('data/steps-large-training.csv')
# x = df['x']
# y = df['y']

# print(A)
# print(X)
# print(np.dot(A, X))


def sigmoid(x):
    value = 1 / (1 + np.exp(-x))
    return value

def calc(x):
    for i in range(5):
        print(sigmoid(x * 2 ** (i-2)))

def repr(x):
    print(f"x = {x}")
    calc(x)

def activations(x):
    n1 = sigmoid(x * 2 ** -2)
    n2 = sigmoid(x * 2 ** -1)
    n3 = sigmoid(x * 2 ** 0)
    n4 = sigmoid(x * 2 ** 1)
    n5 = sigmoid(x * 2 ** 2)
    return n1, n2, n3, n4, n5
    


for i in range(-2, 3):
    repr(i)

n = 10

import matplotlib.pyplot as plt

xs = np.arange(-3, 3, 0.01)
n1s = []
n2s = []
n3s = []
n4s = []
n5s = []
weights = []

gaps13 = []
gaps35 = []
for x in xs:
    n1, n2, n3, n4, n5 = activations(x)
    n1s.append(n1 - 0.5)
    n2s.append(n2 - 0.5)
    n3s.append(n3 - 0.5)
    n4s.append(n4 - 0.5)
    n5s.append(n5 - 0.5)

    gap13 = n1 - n3
    gap35 = n3 - n5

    gaps13.append(gap13)
    gaps35.append(gap35)

    weights.append(6*n5 - 10*n1)

def plot_lines():
    plt.plot(xs, n1s, label='n1')
    plt.plot(xs, n2s, label='n2')
    plt.plot(xs, n3s, label='n3')
    plt.plot(xs, n4s, label='n4')
    plt.plot(xs, n5s, label='n5')
    plt.xlabel('x')
    plt.legend()
    plt.show()

# plt.plot(xs, gaps13, label='gap 1-3')
# plt.plot(xs, gaps35, label='gap 3-5')

def plot_weights():
    plt.plot(xs, weights, label='weights')
    plt.axhline(y=0, color='k', linestyle='--')
    plt.axvline(x=1.5, color='r', linestyle='--')
    plt.xlabel('x')
    plt.legend()
    plt.show()

# plot_lines()
plot_weights()

