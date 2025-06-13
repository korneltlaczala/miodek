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

    def plot(self, start_age, end_age, smoothing_interval=1, scale="log", ax=None):
        """
        Plots the training and test loss over a range of epochs.
        Args:
            start_age (int): The starting epoch index for plotting.
            end_age (int): The ending epoch index for plotting (exclusive).
            smoothing_interval (int, optional): Interval for smoothing the loss curves by sampling every nth epoch. Defaults to 1 (no smoothing).
            scale (str, optional): The scale for the y-axis. Options are:
                - "linear": Linear scale.
                - "log": Logarithmic scale (default).
            ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, uses current axes.
        Displays:
            A matplotlib plot showing the training and test loss curves for the specified epoch range.
        """
        epochs = np.arange(start_age, end_age, smoothing_interval)
        plot_obj = ax if ax is not None else plt
        if ax is None:
            plot_obj.figure(figsize=(12, 5))
            plot_obj.title(f"{self.model.name} - Epochs: {start_age} - {end_age}")
            plot_obj.yscale(scale)
            plot_obj.xlabel("Epoch")
            plot_obj.ylabel(f"Loss ({self.model.data.loss_function})")
        else:
            plot_obj.set_title(f"{self.model.name} - Epochs: {start_age} - {end_age}")
            plot_obj.set_yscale(scale)
            plot_obj.set_xlabel("Epoch")
            plot_obj.set_ylabel(f"Loss ({self.model.data.loss_function})")
        plot_obj.plot(epochs, self.loss_train[start_age:end_age:smoothing_interval], color="red", linewidth=1)
        plot_obj.plot(epochs, self.loss_test[start_age:end_age:smoothing_interval], color="blue", linewidth=1)
        plot_obj.grid(True)
        plot_obj.legend(["Train loss", "Test loss"])