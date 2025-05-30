from matplotlib.pylab import copy
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from activation_functions import *
from history import ModelHistory
from initializers import *
import copy
import pickle
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

    def get_batches(self, batch_size=None):
        n = self.X_train.shape[0]
        if batch_size is None:
            batch_size = {
                n <= 100: 10,
                100 < n <= 500: 20,
                500 < n <= 1000: 32,
                1000 < n <= 5000: 50,
                5000 < n <= 10000: 70,
            }.get(True, 100)
        indices = np.arange(n)
        np.random.shuffle(indices)
        Xs = np.array_split(self.X_train[indices], n // batch_size)
        ys = np.array_split(self.y_train[indices], n // batch_size)
        return Xs, ys

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
        X_train = self.x_scaler.inverse_transform(self.X_train)[:, 0]
        X_test = self.x_scaler.inverse_transform(self.X_test)[:, 0]
        y_train = self.y_scaler.inverse_transform(self.y_train)
        y_test = self.y_scaler.inverse_transform(self.y_test)
        plt.scatter(X_train, y_train, color="blue")
        plt.scatter(X_test, y_test, color="red")

    def plot(self, X, y_pred):
        self.plot_true()
        X = self.x_scaler.inverse_transform(X)
        y_pred = self.y_scaler.inverse_transform(y_pred)
        plt.plot(X, y_pred, color="green")

    def get_range(self):
        min = np.min(np.concatenate((self.X_train[:, 0], self.X_test[:, 0])))
        max = np.max(np.concatenate((self.X_train[:, 0], self.X_test[:, 0])))
        return min, max

    def get_linspace(self):
        min, max = self.get_range()
        margin = 0.2 * (max - min)
        return np.linspace(min - margin, max + margin, 1000).reshape(-1, 1)

class Layer():

    def __init__(self, neurons_in, neurons_out, activation_func, index=None):
        self.neurons_in = neurons_in
        self.neurons_out = neurons_out
        self.activation_func = activation_func
        self.index = index
        self.gradient_clip_treshold = 1

        self.momentum_weights = np.zeros((neurons_in, neurons_out))
        self.momentum_biases = np.zeros(neurons_out)
        self.rms_weights = np.zeros((neurons_in, neurons_out))
        self.rms_biases = np.zeros(neurons_out)
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

    def update(self, prev_layer, learning_rate, momentum_lambda):
        batch_size = prev_layer.a.shape[0]
        gradient = np.dot(prev_layer.a.T, self.e) / batch_size
        gradient = np.clip(gradient, -self.gradient_clip_treshold, self.gradient_clip_treshold)

        self.momentum_weights = momentum_lambda * self.momentum_weights + learning_rate * gradient
        self.momentum_biases = momentum_lambda * self.momentum_biases + learning_rate * np.mean(self.e, axis=0)

        self.weights -= self.momentum_weights
        self.biases -= self.momentum_biases

    def update_rmsprop(self, prev_layer, learning_rate, rms_beta):
        batch_size = prev_layer.a.shape[0]
        gradient = np.dot(prev_layer.a.T, self.e) / batch_size
        gradient = np.clip(gradient, -self.gradient_clip_treshold, self.gradient_clip_treshold)

        self.rms_weights = rms_beta * self.rms_weights + (1 - rms_beta) * gradient ** 2
        self.rms_biases = rms_beta * self.rms_biases + (1 - rms_beta) * np.mean(self.e, axis=0) ** 2

        epsilon = 1e-8
        self.weights -= learning_rate * gradient / (np.sqrt(self.rms_weights) + epsilon)
        self.biases -= learning_rate * np.mean(self.e, axis=0) / (np.sqrt(self.rms_biases) + epsilon)

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
    def update(self, prev_layer, learning_rate, momentum_lambda):
        batch_size = prev_layer.shape[0]
        gradient = np.dot(prev_layer.T, self.e) / batch_size
        gradient = np.clip(gradient, -self.gradient_clip_treshold, self.gradient_clip_treshold)

        self.momentum_weights = momentum_lambda * self.momentum_weights + learning_rate * gradient
        self.momentum_biases = momentum_lambda * self.momentum_biases + learning_rate * np.mean(self.e, axis=0)

        self.weights -= self.momentum_weights
        self.biases -= self.momentum_biases

    def update_rmsprop(self, prev_layer, learning_rate, rms_beta):
        batch_size = prev_layer.shape[0]
        gradient = np.dot(prev_layer.T, self.e) / batch_size
        gradient = np.clip(gradient, -self.gradient_clip_treshold, self.gradient_clip_treshold)

        self.rms_weights = rms_beta * self.rms_weights + (1 - rms_beta) * gradient ** 2
        self.rms_biases = rms_beta * self.rms_biases + (1 - rms_beta) * np.mean(self.e, axis=0) ** 2

        epsilon = 1e-8
        self.weights -= learning_rate * gradient / (np.sqrt(self.rms_weights) + epsilon)
        self.biases -= learning_rate * np.mean(self.e, axis=0) / (np.sqrt(self.rms_biases) + epsilon)


class LastLayer(Layer):
    def backward(self, y_true):
        errors = self.a - y_true
        self.e = errors * self.activation_func.derivative(self.z)


class MLP():

    def __init__(self, architecture, dataset_name, initializer=None, activation_func=None, name="model"):
        self.architecture = architecture
        self.data = DataSet(dataset_name)
        self.age = 0
        self.name = name

        self.initializer = initializer
        self.history = ModelHistory(self)
        self.set_activation_func(activation_func)
        self.generate_layers()
        self.initialize()

    def print_shit(self):
        print("shit")

    def set_activation_func(self, activation_func):
        if activation_func is None:
            activation_func = Sigmoid()
        self.activation_func = activation_func
        self.last_layer_activation_func = Linear()

    def initialize(self):
        if self.initializer is None:
            self.initializer = NormalInitializer()
        self.initializer.initialize(self)

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
        

    def get_weights(self):
        weights = []
        for layer in self.layers:
            weights.append(layer.weights)
        return weights

    def get_biases(self):
        biases = []
        for layer in self.layers:
            biases.append(layer.biases)
        return biases

    def get_weights_copy(self):
        weights = self.get_weights()
        return [w.copy() for w in weights]

    def get_biases_copy(self):
        biases = self.get_biases()
        return [b.copy() for b in biases]

    def set_weights(self, weights):
        for i in range(len(weights)):
            self.layers[i].weights = weights[i]

    def set_biases(self, biases):
        for i in range(len(biases)):
            self.layers[i].biases = biases[i]

    def _forward(self, X, save):
        for layer in self.layers:
            X = layer.forward(X, save=save)
        if save:
            self.y_pred = X
        return X

    def predict(self, X):
        return self._forward(X=X, save=False)

    def predict_train(self):
        X = self.data.X_train
        return self._forward(X=X, save=False)

    def predict_test(self):
        X = self.data.X_test
        return self._forward(X=X, save=False)

    def backprop(self, X, y_true, learning_rate, optimizer, momentum_lambda=None, rms_beta=None):
        self._forward(X=X, save=True)
        self.backward(y_true)
        self.update(X, learning_rate, optimizer, momentum_lambda, rms_beta)

    def backward(self, y_true):
        next_layer = y_true
        for layer in reversed(self.layers):
            layer.backward(next_layer)
            next_layer = layer

    def update(self, X, learning_rate, optimizer, momentum_lambda, rms_beta):
        prev_layer = X
        for layer in self.layers:
            if optimizer == "momentum":
                layer.update(prev_layer, learning_rate, momentum_lambda)
            elif optimizer == "rmsprop":
                layer.update_rmsprop(prev_layer, learning_rate, rms_beta)
            prev_layer = layer

    def train(self, epochs, learning_rate, batch=False, batch_size=None, verbose=True, report_interval=100, momentum_lambda=0, rms_beta=None, optimizer="momentum", save_till_best=False):
        for epoch in range(epochs):
            self.age += 1
            if not batch:
                X = self.data.X_train
                y_true = self.data.y_train
                self.backprop(X, y_true, learning_rate, optimizer, momentum_lambda, rms_beta)
            else:
                Xs, ys = self.data.get_batches(batch_size)
                for X, y in zip(Xs, ys):
                    self.backprop(X, y, learning_rate, optimizer, momentum_lambda, rms_beta)

            self.history.log()
            if verbose and (epoch + 1) % report_interval == 0:
                print(f"Model age: {self.age}", end="\t")
                self.evaluate()

        if save_till_best:
            self.revert_to_best()

    def revert_to_best(self):
        print("-" * 20)
        print(f"Reverting to best model at age {self.history.best_age}")
        print(f"Best model MSE: {round(self.history.best_test_mse, 2)}")
        print("-" * 20)
        self.age = self.history.best_age
        best_weights = copy.deepcopy(self.history.best_weights)
        best_biases = copy.deepcopy(self.history.best_biases)
        self.set_weights(best_weights)
        self.set_biases(best_biases)

    def evaluate(self, evaluate_on_test=True):
        y_pred_train = self.predict_train()
        y_pred_test = self.predict_test()
        mse_train = self.data.evaluate_train(y_pred_train)
        mse_test = self.data.evaluate_test(y_pred_test)
        print(f"MSE on train set: {round(mse_train, 2)}", end="\t")
        print(f"MSE on test set: {round(mse_test, 2)}")

    def plot(self):
        X = self.data.get_linspace()
        y_pred = self.predict(X = X)
        self.data.plot(X, y_pred)
        plt.show()

    def plot_history(self, start_age = 0, end_age = None):
        if end_age is None:
            end_age = self.age
        self.history.plot(start_age, end_age)
        plt.show()

    def copy(self):
        return copy.deepcopy(self)

    def save(self, name="model"):
        with open(f"{name}.pkl", "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(self, name="model"):
        with open(f"{name}.pkl", "rb") as f:
            return pickle.load(f)


class ModelSet():
    def __init__(self, architecture, dataset_name, initializer=None, activation_func=None):
        self.model_full = MLP(architecture=architecture, dataset_name=dataset_name, initializer=initializer, activation_func=activation_func, name="full model")
        self.model_minibatch = MLP(architecture=architecture, dataset_name=dataset_name, initializer=initializer, activation_func=activation_func, name = "mini-batch model")

    def train(self, epochs, learning_rate):
        self.model_full.train(epochs=epochs, learning_rate=learning_rate)
        self.model_minibatch.train(epochs=epochs, learning_rate=learning_rate, batch=True)

    def plot(self):
        pass

if __name__ == "__main__":
    run.main()