import matplotlib.pyplot as plt
import numpy as np

class ModelHistory():
    def __init__(self, model):
        self.model = model
        self.loss_train = []
        self.loss_test = []
        self.data_indices = []
        self.weight_data = []
        self.bias_data = []
        self.save_interval = 30

        self.best_loss_test = np.inf
        
    def log(self):
        y_pred_train = self.model.predict_train()
        y_pred_test = self.model.predict_test()
        loss_train = self.model.data.evaluate_train(y_pred_train)
        loss_test = self.model.data.evaluate_test(y_pred_test)

        if loss_test < self.best_loss_test:
            self.best_loss_test = loss_test
            self.best_age = self.model.age
            self.best_weights = self.model.get_weights()
            self.best_biases = self.model.get_biases()

        self.loss_train.append(loss_train)
        self.loss_test.append(loss_test)
        # if self.model.age % self.save_interval == 0:
        #     self.weight_data.append(self.model.get_weights())
        #     self.bias_data.append(self.model.get_biases())

    def cutoff_till_best(self):
        """
        Cuts off the training history to only include data up to the best epoch.
        This is useful for plotting or analyzing the model's performance up to its best point.
        """
        if hasattr(self, 'best_age'):
            self.loss_train = self.loss_train[:self.best_age]
            self.loss_test = self.loss_test[:self.best_age]
            # self.weight_data = self.weight_data[:self.best_age]
            # self.bias_data = self.bias_data[:self.best_age]
        else:
            print("No best epoch found. Ensure that log() has been called at least once.")

    def plot(self, start_age, end_age, smoothing_interval=1, scale="log", final=True):
        """
        Plots the training and test loss over a range of epochs.
        Args:
            start_age (int): The starting epoch index for plotting.
            end_age (int): The ending epoch index for plotting (exclusive).
            smoothing_interval (int, optional): Interval for smoothing the loss curves by sampling every nth epoch. Defaults to 1 (no smoothing).
            scale (str, optional): The scale for the y-axis. Options are:
                - "linear": Linear scale.
                - "log": Logarithmic scale (default).
        Displays:
            A matplotlib plot showing the training and test loss curves for the specified epoch range.
        """
        epochs = np.arange(start_age, end_age, smoothing_interval)
        if final:
            plt.figure(figsize=(12, 5))
        plt.title(f"{self.model.name} - Epochs: {start_age} - {end_age}")
        plt.plot(epochs, self.loss_train[start_age:end_age:smoothing_interval], color="red", linewidth=1)
        plt.plot(epochs, self.loss_test[start_age:end_age:smoothing_interval], color="blue", linewidth=1)
        plt.yscale(scale)
        plt.grid(True)
        plt.xlabel("Epoch")
        plt.ylabel(f"Loss ({self.model.data.loss_function})")
        plt.legend(["Train loss", "Test loss"])