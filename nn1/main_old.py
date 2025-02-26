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
        self.randomize()

    def generate_layers(self):
        self.layers = []
        self.layers.append(Layer(self.inputs, self.neuron_counts[0]))
        for i in range(1, len(self.neuron_counts)):
            self.layers.append(Layer(self.neuron_counts[i - 1], self.neuron_counts[i]))
        self.layers.append(LastLayer(self.neuron_counts[-1], self.outputs))

    def randomize(self):
        for layer in self.layers:
            layer.draw_weights()
            layer.draw_bias()
 
    def forward(self, input):
        for layer in self.layers:
            output = layer.activate(input)
            input = output
        self.output = output
        return output

    def save(self, name):
        with open(f'{name}.pkl', 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
    

class Layer():

    def __init__(self, neurons_in, neurons_out):
        self.neurons_in = neurons_in
        self.neurons_out = neurons_out
        self.weights = np.zeros((neurons_in, neurons_out))
        self.bias = np.zeros(neurons_out)

    def draw_weights(self):
        self.weights = np.random.uniform(-100, 100, (self.neurons_in, self.neurons_out))

    def draw_bias(self):
        self.bias = np.random.uniform(-100, 100, self.neurons_out)

    def show_weights(self):
        print(str(self.weights))

    def show_bias(self):
        print(str(self.bias))

    def activate(self, input):
        self.output = np.dot(input, self.weights) + self.bias
        self.output = 1 / (1 + np.exp(-self.output))
        return self.output


class LastLayer(Layer):

    def activate(self, input):
        self.output = np.dot(input, self.weights) + self.bias
        return self.output
        
def test2():
    model = MLP(1,[5],1)
    print(model.forward(-2))
    print(model.forward(-1))
    print(model.forward(0))
    print(model.forward(1))
    print(model.forward(2))


def test():
    layer = Layer(neurons_in=2, neurons_out=5)
    layer.draw_weights()
    layer.draw_bias()
    layer.show_weights()
    layer.show_bias()
    print(layer.activate([1, 2]))

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

tester = Tester()
print(tester.best_evaluation)