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

    def activate(self, X):
        for layer in self.layers:
            X = layer.activate(X)
        return X

    def forward(self, X):
        pass

    def save(self, name):
        with open(f'{name}.pkl', 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def load(self, name):
        with open(f'{name}.pkl', 'rb') as f:
            self = pickle.load(f)

class Layer():

    def __init__(self, neurons_in, neurons_out, index=None):
        self.neurons_in = neurons_in
        self.neurons_out = neurons_out
        self.weights = np.zeros((neurons_in, neurons_out))
        self.bias = np.zeros(neurons_out)
        self.index = index

    def activate(self, input):
        self.output = np.dot(input, self.weights) + self.bias
        self.output = 1 / (1 + np.exp(-self.output))
        return self.output

    @property
    def name(self):
        return f"Layer {self.index}"

    def __str__(self):
        output =  f"{self.name}: {self.neurons_in} -> {self.neurons_out}"
        for i in range(self.neurons_out):
            output += f"\n\tNeuron {i}: "
            output += f"\n\t\tWeights: {self.weights[:, i]}"
            output += f"\n\t\tBias: {self.bias[i]}"
        return output
            

class LastLayer(Layer):

    def activate(self, input):
        self.output = np.dot(input, self.weights) + self.bias
        return self.output
        
class Tester():

    def __init__(self, n=2):
        self.n = n
        filename1 = "data/steps-large-training.csv"
        filename2 = "data/square-simple-training.csv"
        self.read_data(filename1)
        self.best_model = MLP(1, [5], 1)
        self.best_evaluation = self.evaluate_model(self.best_model)
        self.search_for_models()
        
    def read_data(self, filename):
        self.df = pd.read_csv(filename)
        self.x = self.df["x"]
        self.y = self.df["y"]

    def evaluate_model(self, model):
        y_hat = np.zeros(len(self.df))
        for index, row in self.df.iterrows():
            x = row["x"]
            y_hat[index] = model.forward([x])[0]
        errors = self.y - y_hat
        mse = np.mean(errors**2)
        return mse
    
    def search_for_models(self):
        for i in range(self.n):
            model = MLP(1, [5], 1)
            model_evaluation = self.evaluate_model(model)
            if model_evaluation < self.best_evaluation:
                self.best_model = model
                self.best_evaluation = model_evaluation
    
# tester = Tester()
# print(tester.best_evaluation)

def run():
    model = MLP(1, [5], 1)
    model.load("model")
    action()
    for layer in model.layers:
        print(layer)
    model.save("model2")

def action():
    pass

if __name__ == "__main__":
    run()