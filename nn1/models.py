import numpy as np
import pickle
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

class MLP():

    def __init__(self, inputs, neuron_counts, outputs):
        self.inputs = inputs
        self.neuron_counts = neuron_counts
        self.outputs = outputs
        self.generate_layers()

    def generate_layers(self):
        self.layers = []
        self.layers.append(Layer(neurons_in=self.inputs, neurons_out=self.neuron_counts[0], index=0))
        for i in range(1, len(self.neuron_counts)):
            self.layers.append(Layer(neurons_in=self.neuron_counts[i - 1], neurons_out=self.neuron_counts[i], index=i))
        self.layers.append(LastLayer(neurons_in=self.neuron_counts[-1], neurons_out=self.outputs, index=len(self.neuron_counts)))

    def forward(self, X):
        for layer in self.layers:
            X = layer.calculate(X)
        self.result = X
        return X

    def save(self, name):
        path = f'models/{name}.pkl'
        with open(f'{name}.pkl', 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def load(self, name):
        with open(f'{name}.pkl', 'rb') as f:
            self = pickle.load(f)
        
    def __str__(self):
        output = ""
        for layer in self.layers:
            output += str(layer)
        return output

class Layer():

    def __init__(self, neurons_in, neurons_out, index=None):
        self.neurons_in = neurons_in
        self.neurons_out = neurons_out
        self.weights = np.zeros((neurons_out, neurons_in))
        self.biases = np.zeros(neurons_out)
        self.index = index

    def activate(self, X):
        X = np.clip(X, -500, 500)
        return 1 / (1 + np.exp(-X))

    def calculate(self, input):
        self.output = np.dot(self.weights, input) + self.biases[:, np.newaxis]
        self.output = self.activate(self.output)
        return self.output

    def set_weights(self, weights):
        self.weights = weights

    def set_biases(self, biases):
        self.biases = biases

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
            

class LastLayer(Layer):

    def calculate(self, input):
        self.output = np.dot(self.weights, input) + self.biases[:, np.newaxis]
        return self.output
        
class Tester():

    def __init__(self, df=None, model=None):
        self.df = df
        self.model = model
        self.ready = False

    def set_df(self, df):
        if self.df is df:
            return
        self.df = df
        self.ready = False

    def set_model(self, model):
        if self.model is model:
            return
        self.model = model
        self.ready = False

    def run(self):
        if self.df is None or self.model is None:
            print("please specify dataset and model first")

        self.x = np.array([self.df["x"]])
        self.y = np.array([self.df["y"]])
        self.y_pred = self.model.forward(self.x)
        self.mse = self.calculate_mse()
        self.ready = True
    
    def calculate_mse(self):
        errors = self.y - self.y_pred
        self.mse = np.mean(errors**2)
        return self.mse

    def report(self):
        if not self.ready:
            self.run()
        print(f"mse: {self.mse}")
