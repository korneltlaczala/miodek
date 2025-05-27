import matplotlib.pyplot as plt
import numpy as np

class ModelHistory():
    def __init__(self, model):
        self.model = model
        self.train_mse = []
        self.test_mse = []
        self.data_indices = []
        self.weight_data = []
        self.bias_data = []
        self.save_interval = 30

        self.best_test_mse = np.inf
        
    def log(self):
        y_pred_train = self.model.predict_train()
        y_pred_test = self.model.predict_test()
        train_mse = self.model.data.evaluate_train(y_pred_train)
        test_mse = self.model.data.evaluate_test(y_pred_test)

        if test_mse < self.best_test_mse:
            self.best_test_mse = test_mse
            self.best_age = self.model.age
            self.best_weights = self.model.get_weights()
            self.best_biases = self.model.get_biases()

        self.train_mse.append(train_mse)
        self.test_mse.append(test_mse)
        if self.model.age % self.save_interval == 0:
            self.weight_data.append(self.model.get_weights())
            self.bias_data.append(self.model.get_biases())

    def plot(self, start_age, end_age):
        smoothing_interval = 30
        epochs = np.arange(start_age, end_age, smoothing_interval)
        plt.title(f"{self.model.name} - Epochs: {start_age} - {end_age}")
        plt.plot(epochs, self.train_mse[start_age:end_age:smoothing_interval], color="blue")
        plt.plot(epochs, self.test_mse[start_age:end_age:smoothing_interval], color="red")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.legend(["Train", "Test"])
        
