import copy
from matplotlib.figure import Figure
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd

from activation_functions import *

class MLP():

    def __init__(self, inputs, neuron_counts, outputs, activation_func=Sigmoid(), activation_func_last_layer=Linear()):
        self.inputs = inputs
        self.neuron_counts = neuron_counts
        self.outputs = outputs

        self.activation_func = activation_func
        self.activation_func_last_layer = activation_func_last_layer

        self.name = f"new_model"
        self.generate_layers()

    def generate_layers(self):
        self.layers = []
        self.layers.append(FirstLayer(neurons_in=self.inputs,
                                      neurons_out=self.neuron_counts[0],
                                      index=0,
                                      activation_func=self.activation_func))
        for i in range(1, len(self.neuron_counts)):
            self.layers.append(Layer(neurons_in=self.neuron_counts[i - 1],
                                     neurons_out=self.neuron_counts[i],
                                     index=i,
                                     activation_func=self.activation_func))
        self.layers.append(LastLayer(neurons_in=self.neuron_counts[-1],
                                     neurons_out=self.outputs,
                                     index=len(self.neuron_counts),
                                     activation_func=self.activation_func_last_layer))

    def forward(self, X, save=True):
        for layer in self.layers:
            X = layer.calculate(X, save=save)
        if save:
            self.result = X
        return X

    def predict(self, X):
        return self.forward(X, save=False)

    def backprop(self, x, y_true, learning_rate):
        self.backward(y_true)
        self.update(x, learning_rate)
    
    def backward(self, y_true):
        next_layer = y_true
        for layer in reversed(self.layers):
            layer.backward(next_layer)
            next_layer = layer

    def update(self, x, learning_rate):
        prev_layer = x
        for layer in self.layers:
            layer.update(prev_layer, learning_rate)
            prev_layer = layer

    def get_copy(self):
        return copy.deepcopy(self)

    def save(self, name=None):
        if name == None:
            name = self.name
        else: 
            self.name = name
        path = f'models/{name}.pkl'
        with open(path, 'wb') as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)
    
    @classmethod
    def load(self, name):
        with open(f'{name}', 'rb') as file:
            return pickle.load(file)
        
    def __str__(self):
        output = ""
        for layer in self.layers:
            output += str(layer)
        return output

class Layer():

    def __init__(self, neurons_in, neurons_out, index, activation_func, gradient_clip_value=2.0):
        self.neurons_in = neurons_in
        self.neurons_out = neurons_out

        self.weights = np.zeros((neurons_out, neurons_in))
        self.biases = np.zeros(neurons_out)

        self.a = None
        self.output = None
        self.deltas = None

        self.index = index
        self.activation_func = activation_func
        self.gcp = gradient_clip_value

    def calculate(self, input, save):
        a = np.dot(self.weights, input) + self.biases[:, np.newaxis]
        output = self.activation_func.activate(a)
        if save:
            self.a = a
            self.output = output
        return output

    def backward(self, next_layer):
        weight_sums = np.dot(next_layer.weights.T, next_layer.deltas)
        self.deltas = weight_sums * self.activation_func.derivative(self.a)

    def update(self, prev_layer, learning_rate):
        clipped_deltas = np.clip(self.deltas, -self.gcp, self.gcp)
        self.weights -= learning_rate * np.dot(clipped_deltas, prev_layer.output.T) / prev_layer.output.shape[1]
        self.biases -= learning_rate * np.mean(clipped_deltas, axis=1)
        # self.weights -= learning_rate * np.clip(np.dot(self.deltas, prev_layer.output.T) / prev_layer.output.shape[1], -self.gcp, self.gcp)
        # self.biases -= learning_rate * np.clip(np.mean(self.deltas, axis=1), -self.gcp, self.gcp)

    def set_weights(self, weights):
        self.weights = weights

    def set_biases(self, biases):
        self.biases = biases

    def get_weights_shape(self):
        return self.weights.shape

    def get_biases_shape(self):
        return self.biases.shape

    @property
    def name(self):
        return f"Layer {self.index}"

    def __str__(self):
        output =  f"{self.name}: {self.neurons_in} -> {self.neurons_out}"
        for i in range(self.neurons_out):
            output += f"\n\tNeuron {i}: "
            output += f"\n\t\tWeights: {self.weights[i, :]}"
            output += f"\n\t\tBias: {self.biases[i]}"
        output += "\n"
        return output
            
class FirstLayer(Layer):
    def update(self, prev_layer, learning_rate):
        clipped_deltas = np.clip(self.deltas, -self.gcp, self.gcp)
        self.weights -= learning_rate * np.dot(clipped_deltas, prev_layer.T) / prev_layer.shape[1]
        self.biases -= learning_rate * np.mean(clipped_deltas, axis=1)
        # self.weights -= learning_rate * np.clip(np.dot(self.deltas, prev_layer.T) / prev_layer.shape[1], -self.gcp, self.gcp)
        # self.biases -= learning_rate * np.clip(np.mean(self.deltas, axis=1), -self.gcp, self.gcp)

class LastLayer(Layer):

    def backward(self, y_true):
        errors = self.output - y_true
        self.deltas = errors * self.activation_func.derivative(self.a)


class Tester():

    def __init__(self, data_file=None, df=None, model=None):
        self.data_file = data_file
        if self.data_file is not None:
            self.df = pd.read_csv(self.data_file)
        else:
            self.df = df
        self.model = model
        self.ready = False

    def set_data_file(self, data_file):
        if self.data_file is data_file:
            return
        self.data_file = data_file
        self.df = pd.read_csv(data_file)
        self.ready = False

    def set_df(self, df):
        if self.df is df:
            return
        self.df = df
        self.data_file = None
        self.ready = False

    def set_model(self, model):
        if self.model is model:
            return
        self.model = model
        self.ready = False

    def run(self):
        if self.ready:
            return True

        if self.df is None or self.model is None:
            self.ready = False
            return False

        self.x = np.array([self.df["x"]])
        self.y = np.array([self.df["y"]])
        self.y_pred = self.model.forward(self.x)
        self.mse = self.calculate_mse()
        self.ready = True
        return True

    def calculate_mse(self):
        errors = self.y - self.y_pred
        self.mse = np.mean(errors**2)
        return self.mse

    def report(self):
        if not self.run():
            return
        print(f"mse: {self.mse}")

    def set_not_ready(self):
        self.ready = False

    def plot(self, linear=False):
        if not self.run():
            return None
        if linear:
            x = np.array([np.linspace(self.x.min(), self.x.max(), 1000)])
            y_pred = self.model.predict(x)

            plt.scatter(self.x, self.y, label="y", s=10, alpha=0.9)
            plt.plot(x.T, y_pred.T, label="y_pred", color='r', alpha=0.8)
            plt.legend()
            return plt

        plt.scatter(self.x, self.y, label="y", s=10, alpha=0.9)
        plt.scatter(self.x, self.y_pred, label="y_pred", s=3, alpha=0.9)
        plt.legend()
        return plt
