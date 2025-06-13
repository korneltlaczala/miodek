import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from activation_functions import *
from history import ModelHistory
from initializers import *
import copy
from tqdm import tqdm, trange

class MLPArchitecture():

    def __init__(self, inputs, layers, outputs):
        self.inputs = inputs
        self.layers = layers
        self.outputs = outputs

    def __str__(self):
        return f"Architecture:\n{self.inputs} -> {self.layers} -> {self.outputs}"

class DataSet():

    def __init__(self, name, loss_function="mse"):
        self.name = name
        self.loss_function = loss_function
        self.read_data()
        self.scale_data()

    def read_data(self):
        train_set_name = f"{self.name}-training.csv"
        test_set_name = f"{self.name}-test.csv"

        self.train_df = self.prep_df(train_set_name)
        self.X_train = self.train_df.iloc[:, :-1].values
        self.y_train = self.train_df.iloc[:, -1].values

        self.test_df = self.prep_df(test_set_name)
        self.X_test = self.test_df.iloc[:, :-1].values
        self.y_test = self.test_df.iloc[:, -1].values

        if self.loss_function == "cross_entropy":
            self.y_train = self.y_train.astype(int)
            self.y_test = self.y_test.astype(int)

            self.n_classes = len(set(self.y_train) | set(self.y_test))

            y_train_onehot = np.zeros((self.y_train.shape[0], self.n_classes))
            for i in range(self.y_train.shape[0]):
                y_train_onehot[i, self.y_train[i]] = 1

            y_test_onehot = np.zeros((self.y_test.shape[0], self.n_classes))
            for i in range(self.y_test.shape[0]):
                y_test_onehot[i, self.y_test[i]] = 1
            
            self.y_train = y_train_onehot
            self.y_test = y_test_onehot

    def scale_data(self):
        if self.loss_function == "cross_entropy":
            class DummyScaler:
                def fit_transform(self, x): return x
                def transform(self, x): return x
                def inverse_transform(self, x): return x
            self.x_scaler = DummyScaler()
            self.y_scaler = DummyScaler()
            return
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
        if self.loss_function == "cross_entropy":
            epsilon = 1e-15
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

            # print(y_true)
            # print(y_pred)
            return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

        y_true_denormalized = self.y_scaler.inverse_transform(y_true)
        y_pred_denormalized = self.y_scaler.inverse_transform(y_pred)
        if self.loss_function == "mse":
            return np.mean((y_true_denormalized - y_pred_denormalized) ** 2)
        if self.loss_function == "mae":
            return np.mean(np.abs(y_true_denormalized - y_pred_denormalized))

    def plot_true(self):
        X_train = self.x_scaler.inverse_transform(self.X_train)[:, 0]
        X_test = self.x_scaler.inverse_transform(self.X_test)[:, 0]
        y_train = self.y_scaler.inverse_transform(self.y_train)
        y_test = self.y_scaler.inverse_transform(self.y_test)
        plt.scatter(X_train, y_train, color="blue")
        plt.scatter(X_test, y_test, color="red")

    def plot_classification(self, data="test_data"):
        if data == "training_data":
            X = self.X_train
            y = self.y_train
        elif data == "test_data":
            X = self.X_test
            y = self.y_test
        else:
            raise ValueError("data must be 'training_data' or 'test_data'")

        y = np.argmax(y, axis=1) if self.loss_function == "cross_entropy" else y
        classes = np.unique(y)
        for cls in classes:
            idx = y == cls
            plt.scatter(
                X[idx, 0],
                X[idx, 1],
                c=[plt.cm.viridis(cls / (len(classes)-1))],
                label=f"Class {cls}",
                s=20
            )
        return plt

    def plot_predicted_classification(self, y_pred, mark_incorrect=True):
        y_pred = np.argmax(y_pred, axis=1) if self.loss_function == "cross_entropy" else y_pred
        classes = np.unique(y_pred)
        for cls in classes:
            idx = y_pred == cls
            plt.scatter(
                self.X_test[idx, 0],
                self.X_test[idx, 1],
                c=[plt.cm.viridis(cls / (len(classes)-1))],
                label=f"Class {cls}",
                s=20
            )
        if mark_incorrect:
            incorrect = y_pred != np.argmax(self.y_test, axis=1)
            plt.scatter(self.X_test[incorrect, 0], self.X_test[incorrect, 1], facecolors='none', edgecolors='red', s=5, label='Incorrect')
        return plt

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
        # print(self.a)
        # print(y_true)
        errors = self.a - y_true
        self.e = errors * self.activation_func.derivative(self.z)


class MLP():

    def __init__(
            self,
            architecture,
            dataset_name,
            loss_function="mse",
            initializer=None,
            activation_func=None,
            last_layer_activation_func=None,
            target_precision=1e-2,
            name="model",
    ):
        self.architecture = architecture
        self.data = DataSet(dataset_name, loss_function)

        self.initializer = initializer
        self.history = ModelHistory(self)
        self.set_activation_func(activation_func, last_layer_activation_func)
        self.generate_layers()
        self.initialize()

        self.target_precision = target_precision
        self.age = 0
        self.epochs_lost = 0
        self.name = name

    def print_shit(self):
        print("shit")

    def set_activation_func(self, activation_func, last_layer_activation_func):
        if activation_func is None:
            activation_func = Sigmoid()
        if last_layer_activation_func is None:
            last_layer_activation_func = Linear()
        self.activation_func = activation_func
        self.last_layer_activation_func = last_layer_activation_func

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
            weights.append(copy.deepcopy(layer.weights))
        return weights

    def get_biases(self):
        biases = []
        for layer in self.layers:
            biases.append(copy.deepcopy(layer.biases))
        return biases

    def set_weights(self, weights):
        for i in range(len(weights)):
            self.layers[i].weights = copy.deepcopy(weights[i])

    def set_biases(self, biases):
        for i in range(len(biases)):
            self.layers[i].biases = copy.deepcopy(biases[i])

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

    def train(self, epochs, learning_rate, batch=False, batch_size=None, verbose=True, momentum_lambda=0, rms_beta=None, optimizer="momentum", save_till_best=False):


        iterator = (
            trange(epochs) if verbose else
            range(epochs)
        )

        start_age = self.age
        for epoch in iterator:
            self.age += 1
            best_loss_test = round(self.history.best_loss_test, self.precision_int)
            loss_test = round(self.data.evaluate_test(self.predict_test()), self.precision_int)
            # if best_loss_test == loss_test:
            #     print(f"new best model at age {self.age-1} with {loss_test} MSE on test set")
            iterator.set_description(
                f"{str(f'Training from age {start_age}: (best_loss_test: {best_loss_test}, loss_test: {loss_test})'):<61}"
            )
            if not batch:
                X = self.data.X_train
                y_true = self.data.y_train
                self.backprop(X, y_true, learning_rate, optimizer, momentum_lambda, rms_beta)
            else:
                Xs, ys = self.data.get_batches(batch_size)
                for X, y in zip(Xs, ys):
                    self.backprop(X, y, learning_rate, optimizer, momentum_lambda, rms_beta)
            self.history.log()

        if save_till_best:
            self.revert_to_best()

        test_loss = round(self.data.evaluate_test(self.predict_test()), self.precision_int)
        print(f"Model training finished at age {self.age} with loss {test_loss} on test set")

    def revert_to_best(self):
        print("-" * 20)
        print(f"Reverting to best model at age {self.history.best_age}")
        self.epochs_lost += self.age - self.history.best_age
        self.age = self.history.best_age
        self.set_weights(self.history.best_weights)
        self.set_biases(self.history.best_biases)
        self.evaluate()
        print("-" * 20)

    def evaluate(self, evaluate_on_test=True):
        y_pred_train = self.predict_train()
        y_pred_test = self.predict_test()
        train_loss = self.data.evaluate_train(y_pred_train)
        test_loss = self.data.evaluate_test(y_pred_test)
        print(f"Loss on train set: {round(train_loss, self.precision_int)}", end="\t")
        print(f"Loss on test set: {round(test_loss, self.precision_int)}")

    def plot(self):
        X = self.data.get_linspace()
        y_pred = self.predict(X = X)
        self.data.plot(X, y_pred)
        plt.show()

    def plot_history(self, start_age=0, end_age=None, smoothing_interval=30, scale="linear"):
        if end_age is None:
            end_age = self.age
        self.history.plot(start_age, end_age, smoothing_interval=smoothing_interval, scale=scale)
        plt.show()

    def plot_classification(self):
        self.data.plot_classification()

    def plot_predicted_classification(self, y_pred=None):
        if y_pred is None:
            y_pred = self.predict_test()
        self.data.plot_predicted_classification(y_pred)
        plt.show()

    def plot_classification_comparison(self, y_pred=None):
        if y_pred is None:
            y_pred = self.predict_test()
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        self.data.plot_classification()
        plt.title("True Classification")
        plt.subplot(1, 2, 2)
        self.data.plot_predicted_classification(y_pred)
        plt.title("Predicted Classification")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()

        self.classification_performance_summary()

    def plot_classification_data(self):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        self.data.plot_classification(data="training_data")
        plt.title("Training data")
        plt.subplot(1, 2, 2)
        self.data.plot_classification(data="test_data")
        plt.title("Test data")
        plt.show()

    def copy(self):
        return copy.deepcopy(self)

    def classification_performance_summary(self):
        print(f"Model: {self.name}")
        print(f"Age: {self.age}")
        print(f"Train Loss: {round(self.data.evaluate_train(self.predict_train()), self.precision_int)}")
        print(f"Test Loss: {round(self.data.evaluate_test(self.predict_test()), self.precision_int)}")
        print(f"Accuracy: {self.accuracy:.4g}")
        print(f"F1 Score: {self.f1_score:.4g}")

    @property
    def precision_int(self):
        return int(-np.log10(self.target_precision))

    @property
    def accuracy(self):
        y_pred = self.predict_test()
        y_true = np.argmax(self.data.y_test, axis=1) if self.data.loss_function == "cross_entropy" else self.data.y_test
        y_pred = np.argmax(y_pred, axis=1) if self.data.loss_function == "cross_entropy" else y_pred
        return accuracy_score(y_true, y_pred)

    @property
    def f1_score(self):
        y_pred = self.predict_test()
        y_true = np.argmax(self.data.y_test, axis=1) if self.data.loss_function == "cross_entropy" else self.data.y_test
        y_pred = np.argmax(y_pred, axis=1) if self.data.loss_function == "cross_entropy" else y_pred
        return f1_score(y_true, y_pred, average='weighted')


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
    pass