import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


class MLP:
    def __init__(self, layers, activation=sigmoid):
        self.activation = activation
        self.layers = layers
        self.weights = [np.random.uniform(-100, 100, (self.layers[i], self.layers[i + 1])) for i in
                        range(len(self.layers) - 1)]
        self.biases = [np.random.uniform(-100, 100, (1, self.layers[i + 1])) for i in range(len(self.layers) - 1)]

    def forward(self, X):
        a = X
        # print(a)
        for i in range(len(self.weights) - 1):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            a = self.activation(z)
        z = np.dot(a, self.weights[-1]) + self.biases[-1]
        return z


# # Wczytanie danych
# square_simple_training = pd.read_csv("C:/studia/3rok/MiO/NN1/dane/square-simple-training.csv", header=None,
#                                      skiprows=1).astype(float)
# square_simple_test = pd.read_csv("C:/studia/3rok/MiO/NN1/dane/square-simple-test.csv", header=None, skiprows=1).astype(
#     float)
# steps_large_training = pd.read_csv("C:/studia/3rok/MiO/NN1/dane/steps-large-training.csv", header=None,
#                                    skiprows=1).astype(float)
# steps_large_test = pd.read_csv("C:/studia/3rok/MiO/NN1/dane/steps-large-test.csv", header=None, skiprows=1).astype(
#     float)
square_simple_training = pd.read_csv("data/square-simple-training.csv", header=None, skiprows=1).astype(float)
square_simple_test = pd.read_csv("data/square-simple-test.csv", header=None, skiprows=1).astype(float)
steps_large_training = pd.read_csv("data/steps-large-training.csv", header=None, skiprows=1).astype(float)
steps_large_test = pd.read_csv("data/steps-large-test.csv", header=None, skiprows=1).astype(float)

X_train = square_simple_training.iloc[:, 1].values.reshape(-1, 1)
y_train = square_simple_training.iloc[:, 2].values.reshape(-1, 1)
X_test = square_simple_test.iloc[:, 1].values.reshape(-1, 1)
y_test = square_simple_test.iloc[:, 2].values.reshape(-1, 1)

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train)
X_test_scaled = scaler_X.transform(X_test)
y_test_scaled = scaler_y.transform(y_test)

X_train2 = steps_large_training.iloc[:, 1].values.reshape(-1, 1)
y_train2 = steps_large_training.iloc[:, 2].values.reshape(-1, 1)
X_test2 = steps_large_test.iloc[:, 1].values.reshape(-1, 1)
y_test2 = steps_large_test.iloc[:, 2].values.reshape(-1, 1)
scaler_X2 = MinMaxScaler()
scaler_y2 = MinMaxScaler()
X_train_scaled2 = scaler_X2.fit_transform(X_train2)
y_train_scaled2 = scaler_y2.fit_transform(y_train2)
X_test_scaled2 = scaler_X2.transform(X_test2)
y_test_scaled2 = scaler_y2.transform(y_test2)



# Testowanie różnych architektur
architectures = [[1, 5, 1], [1, 10, 1], [1, 5, 5, 1]]
# architectures = [[1,4,3,3,1]]


def best_mse_finder(architectures):
    best_mse = float('inf')
    best_model = None

    for arch in architectures:
        for _ in range(1000):
            model = MLP(layers=arch, activation=sigmoid)
            y_pred_scaled = model.forward(X_train_scaled)
            y_pred = scaler_y.inverse_transform(y_pred_scaled)
            error = mse(y_train, y_pred)

            if error < best_mse:
                best_mse = error
                best_model = model

    print(f'Najlepsze MSE na zbiorze treningowym: {best_mse}\n'
          f'Dla modelu: {best_model.layers}\n'
          f'Wagi: {best_model.weights}\n')

    # Testowanie na zbiorze testowym
    y_test_pred_scaled = best_model.forward(X_test_scaled)
    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled)
    test_mse = mse(y_test, y_test_pred)
    print(f'MSE na zbiorze testowym: {test_mse}')


# best_mse_finder(architectures)

def best_mse_finder2(architectures):
    best_mse2 = float('inf')
    best_model2 = None

    for arch in architectures:
        for _ in range(100):
            model = MLP(layers=arch, activation=sigmoid)
            y_pred_scaled2 = model.forward(X_train_scaled2)
            # print(y_pred_scaled2)
            y_pred2 = scaler_y2.inverse_transform(y_pred_scaled2)
            error2 = mse(y_train2, y_pred2)

            if error2 < best_mse2:
                best_mse2 = error2
                best_model2 = model

    print(f'Najlepsze MSE na zbiorze treningowym: {best_mse2}\n'
          f'Dla modelu: {best_model2.layers}\n'
          f'Wagi: {best_model2.weights}\n')

    # Testowanie na zbiorze testowym
    y_test_pred_scaled2 = best_model2.forward(X_test_scaled2)
    y_test_pred2 = scaler_y2.inverse_transform(y_test_pred_scaled2)
    test_mse2 = mse(y_test2, y_test_pred2)
    print(f'MSE na zbiorze testowym: {test_mse2}')


# best_mse_finder2(architectures)
model = MLP(1, 5, 1)
