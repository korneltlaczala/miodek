import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from activation_functions import *
import run

class MLPArchitecture():

    def __init__(self, inputs, layers, outputs):
        self.inputs = inputs
        self.layers = layers
        self.outputs = outputs

    def __str__(self):
        return f"Architecture:\n{self.inputs} -> {self.layers} -> {self.outputs}"

class DataSet():

    def __init__(self, name):
        self.name = name
        self.read_data()
        self.scale_data()

    def read_data(self):
        train_set_name = f"{self.name}-training.csv"
        test_set_name = f"{self.name}-test.csv"

        self.train_df = self.prep_df(train_set_name)
        self.test_df = self.prep_df(test_set_name)

        self.X_train = self.train_df.iloc[:, :-1].values
        self.y_train = self.train_df.iloc[:, -1].values

        self.X_test = self.test_df.iloc[:, :-1].values
        self.y_test = self.test_df.iloc[:, -1].values

    def scale_data(self):
        self.x_scaler = StandardScaler()
        self.X_train = self.x_scaler.fit_transform(self.X_train)
        self.X_test = self.x_scaler.transform(self.X_test)

        self.y_scaler = StandardScaler()
        self.y_train = self.y_scaler.fit_transform(self.y_train.reshape(-1, 1))
        self.y_test = self.y_scaler.transform(self.y_test.reshape(-1,1))
        
    def prep_df(self, filename):
        df = pd.read_csv(f"data/{filename}")
        df = df.drop(columns=["Unnamed: 0", "id"], errors="ignore")
        return df

    def evaluate_train(self, y_pred_train):
        return self._evaluate(self.y_train, y_pred_train)

    def evaluate_test(self, y_pred_test):
        return self._evaluate(self.y_test, y_pred_test)

    def _evaluate(self, y_true, y_pred):
        y_true_denormalized = self.y_scaler.inverse_transform(y_true)
        y_pred_denormalized = self.y_scaler.inverse_transform(y_pred)
        mse = np.mean((y_true_denormalized - y_pred_denormalized) ** 2)
        return mse

    def plot_true(self):
        plt.scatter(self.X_train[:, 0], self.y_train, color="blue")
        plt.scatter(self.X_test[:, 0], self.y_test, color="red")
        plt.show()

    def plot_test(self, y_pred_test):
        plt.scatter(self.X_test[:, 0], self.y_test, color="blue")
        plt.scatter(self.X_test[:, 0], y_pred_test, color="red")
        plt.show()


class Layer():

    def __init__(self, neurons_in, neurons_out, activation_func, index=None):
        self.neurons_in = neurons_in
        self.neurons_out = neurons_out
        self.activation_func = activation_func
        self.index = index
        self.gradient_clip_treshold = 1

        self.weights = np.zeros((neurons_in, neurons_out))
        self.biases = np.zeros(neurons_out)

        self.z = None
        self.a = None
        self.errors = None

    def forward(self, X, save):
        z = np.dot(X, self.weights) + self.biases
        a = self.activation_func.activate(z)
        if save:
            self.z = z
            self.a = a
        return a

    def backward(self, next_layer):
        weight_sums = np.dot(next_layer.e, next_layer.weights.T)
        self.e = weight_sums * self.activation_func.derivative(self.z)

    def update(self, prev_layer, learning_rate):
        batch_size = prev_layer.a.shape[0]
        gradient = np.dot(prev_layer.a.T, self.e) / batch_size
        self.weights -= learning_rate * gradient
        self.biases -= learning_rate * np.mean(self.e, axis=0)

    def set_weigths(self, weights):
        self.weights = weights
    
    def set_biases(self, biases):
        self.biases = biases

    @property
    def name(self):
        return f"Layer {self.index}"

    def __str__(self):
        output = f"{self.name}: {self.neurons_in} -> {self.neurons_out}\n"
        for i in range(self.neurons_out):
            output += f"\tNeuron {i}:"
            output += f"\t\tWeights: {self.weights[:, i]}"
            output += f"\t\tBias: {self.biases[i]}"
        output += "\n"
        return output


class FirstLayer(Layer):
    def update(self, prev_layer, learning_rate):
        batch_size = prev_layer.shape[0]
        gradient = np.dot(prev_layer.T, self.e) / batch_size
        self.weights -= learning_rate * gradient
        self.biases -= learning_rate * np.mean(self.e, axis=0)


class LastLayer(Layer):
    def backward(self, y_true):
        errors = self.a - y_true
        self.e = errors * self.activation_func.derivative(self.z)


class MLP():

    def __init__(self, architecture, dataset_name):
        self.architecture = architecture
        self.data = DataSet(dataset_name)

        self.activation_func = Sigmoid()
        self.last_layer_activation_func = Linear()
        self.generate_layers()

    def generate_layers(self):
        self.layers = []
        self.layers.append(FirstLayer(neurons_in=self.architecture.inputs,
                                      neurons_out=self.architecture.layers[0],
                                      activation_func=self.activation_func,
                                      index=0))
        for i in range(1, len(self.architecture.layers)):
            self.layers.append(Layer(neurons_in=self.architecture.layers[i-1],
                                     neurons_out=self.architecture.layers[i],
                                     activation_func=self.activation_func,
                                     index=i))
        self.layers.append(LastLayer(neurons_in=self.architecture.layers[-1],
                                     neurons_out=self.architecture.outputs,
                                     activation_func=self.last_layer_activation_func,
                                     index=len(self.architecture.layers)))

    def _forward(self, X, save):
        for layer in self.layers:
            X = layer.forward(X, save=save)
        if save:
            self.y_pred = X
        return X

    def forward_train(self):
        X = self.data.X_train
        return self._forward(X=X, save=False)

    def predict_train(self):
        X = self.data.X_train
        return self._forward(X=X, save=False)

    def predict_test(self):
        X = self.data.X_test
        return self._forward(X=X, save=False)

    def backprop(self, X, y_true, learning_rate):
        self._forward(X=X, save=True)
        self.backward(y_true)
        self.update(X, learning_rate)

    def backward(self, y_true):
        next_layer = y_true
        for layer in reversed(self.layers):
            layer.backward(next_layer)
            next_layer = layer

    def update(self, X, learning_rate):
        prev_layer = X
        for layer in self.layers:
            layer.update(prev_layer, learning_rate)
            prev_layer = layer

    def train(self, epochs, learning_rate):
        for epoch in range(epochs):
            X = self.data.X_train
            y_true = self.data.y_train
            self.backprop(X=X, y_true=y_true, learning_rate=learning_rate)
            if (epoch + 1) % 100 == 0:
                print(f"Epoch: {epoch + 1}")
                self.evaluate()

    def evaluate(self):
        y_pred_train = self.predict_train()
        y_pred_test = self.predict_test()
        mse_train = self.data.evaluate_train(y_pred_train)
        mse_test = self.data.evaluate_test(y_pred_test)
        print(f"MSE on train set: {mse_train}")
        print(f"MSE on test set: {mse_test}")

    def plot(self):
        y_pred_test = self.predict_test()
        self.data.plot_test(y_pred_test)


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


if __name__ == "__main__":
    run.main()