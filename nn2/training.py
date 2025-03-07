from abc import abstractmethod

import numpy as np
from models import *

def run():
    # model = MLP(1, [16, 30, 16], 1)
    # data_file = "data/square-simple-test.csv"
    # data_file = "data/multimodal-large-test.csv"
    # trainer = Trainer(model, data_file)
    trainer = Trainer.load("trainer_square_bartek")
    trainer.test()
    trainer.train(epochs=2e5, learning_rate=0.001)
    trainer.ask_for_save("trainer_square_bartek")
    # trainer.ask_for_save("trainer_multimodal")

class Trainer():

    def __init__(self, model, data_file, target_epoch=1e5, name="default_trainer"):
        self.model = model
        self.data_file = data_file
        self.tester = Tester(model=model, data_file=data_file)
        self.current_epoch = 0
        self.target_epoch = target_epoch
        self.name = name
        self.prepare_data()
        self.initialize_model()

    def initialize_model(self):
        initializer = RandomInitializer()
        initializer.initialize(self.model)

    def prepare_data(self):
        self.tester.set_data_file(self.data_file)
        self.df = pd.read_csv(self.data_file)
        self.X = np.array([self.df["x"]])
        self.y = np.array([self.df["y"]])

    def train(self, epochs, learning_rate=0.001, report_interval=1e4):
        self.target_epoch = self.current_epoch + epochs
        self.tester.run()
        self.mse_min = self.tester.mse
        self.report_interval = report_interval
        self.report(starting=True)
        while self.current_epoch < self.target_epoch:
            self.model.backprop(x=self.X, y_true=self.y, learning_rate=learning_rate)
            self.tester.set_not_ready()
            self.tester.run()
            self.current_epoch += 1
            self.report()
            self.save_min_mse()

        self.test()

    def save_min_mse(self):
        if self.tester.mse < self.mse_min:
            self.mse_min = self.tester.mse
            self.save(name=self.name)
            
    def report(self, starting=False):
        if starting:
            self.report_start()
        if self.current_epoch % self.report_interval== 0:
            print(f"epoch: {self.current_epoch}\t\tmse: {round(self.tester.mse, 2)}")

    def report_start(self):
        print("----------------------")
        print(f"Starting training")
        print(f"model age: {self.current_epoch} of {self.target_epoch} epochs")
        print(f"epochs to go: {self.target_epoch - self.current_epoch}")
        print(f"starting mse: {self.tester.mse}")
        print("----------------------")

    def test(self):
        self.tester.report()
        plt = self.tester.plot(linear=True)
        plt.show()

    def save(self, name=None):
        if name == None:
            name = self.name
        else: 
            self.name = name
        path = f'trainers/{name}.pkl'
        with open(path, 'wb') as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)
        print(f"Trainer (age: {self.current_epoch}) saved as {name}.\t mse: {round(self.tester.mse, 2)}")

    def ask_for_save(self, name):
        response = None
        while response not in ['y', 'n']:
            response = input("Save trainer? (y/n) ")
        if response == 'y':
            self.save(name)
        else:
            print("Trainer not saved")
    
    @classmethod
    def load(self, name):
        path = f'trainers/{name}.pkl'
        with open(path, 'rb') as file:
            return pickle.load(file)

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

if __name__ == "__main__":
    run()