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
        self.plot()
        print(f"X_train 1st: {self.X_train[0, 0]}")

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

    def plot(self):
        print(f"X_train shape: {self.X_train.shape}")
        print(f"y_train shape: {self.y_train.shape}")
        plt.scatter(self.X_train[:, 0], self.y_train, color="blue")
        plt.scatter(self.X_test[:, 0], self.y_test, color="red")
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
        print(f"X shape: {X.shape}")
        z = np.dot(X, self.weights) + self.biases
        a = self.activation_func.activate(z)
        if save:
            self.z = z
            self.a = a
        return a

    # def backward(self, next_layer):
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
    def update(self, preve_layer, learning_rate):
        pass


class LastLayer(Layer):
    def backward(self, y_true):
        pass


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

    def forward(self, save=True):
        X = self.data.X_train
        return self._forward(X, save)

    def _forward(self, X, save):
        for layer in self.layers:
            X = layer.forward(X, save=save)
        if save:
            self.y_pred = X
        return X

    def predict(self):
        return self.forward(X, save=False)


if __name__ == "__main__":
    run.main()