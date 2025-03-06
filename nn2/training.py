from abc import abstractmethod

import numpy as np
from models import *

def run():
    model = MLP(1, [5], 1)
    data_file = "data/square-simple-test.csv"
    trainer = Trainer(model, data_file)
    trainer.train()

class Trainer():

    def __init__(self, model, data_file, epochs=10):
        self.model = model
        self.tester = Tester(model=model, data_file=data_file)
        self.current_epoch = 0
        self.epochs = epochs
        self.initialize_model()

    def initialize_model(self):
        initializer = RandomInitializer()
        initializer.initialize(self.model)

    def train(self):
        while self.current_epoch < self.epochs:
            self.test()
            self.current_epoch += 1

    def test(self):
        self.tester.report()
        plt = self.tester.plot()
        plt.show()

class Initializer():

    def __init__(self):
        pass

    def initialize(self, model):
        self.initialize_weights(model)
        self.initialize_biases(model)

    @abstractmethod
    def initialize_weights(self, model):
        pass

    @abstractmethod
    def initialize_biases(self, model):
        pass

class RandomInitializer(Initializer):

    def initialize_weights(self, model):
        for layer in model.layers:
            layer.set_weights(np.random.uniform(0, 1, size=layer.get_weights_shape()))

    def initialize_biases(self, model):
        for layer in model.layers:
            layer.set_biases(np.random.uniform(0, 1, size=layer.get_biases_shape()))

run()