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
        plt.scatter(self.X_train, self.y_train, color="blue")
        plt.scatter(self.X_test, self.y_test, color="red")
        plt.show()

class MLP():

    def __init__(self, architecture, dataset_name):
        self.architecture = architecture
        self.data = DataSet(dataset_name)

        self.activation_func = Sigmoid()
        self.last_layer_activation_func = Linear()

if __name__ == "__main__":
    run.main()