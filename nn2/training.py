from abc import abstractmethod

import numpy as np
from models import *

def run():
    data_file = "data/square-simple-test.csv"
    # trainer_name = "trainer_square_2000"
    # trainer_name = "trainer_square"
    trainer_name = "trainer_app"
    # data_file = "data/multimodal-large-test.csv"
    # trainer_name = "trainer_multimodal"
    # model = MLP(1, [16, 30, 16], 1)
    # trainer = Trainer(model, data_file, name=trainer_name)
    trainer = Trainer.load(trainer_name)
    trainer.test()
    trainer.train(epochs=5e4, learning_rate=0.1, auto_save=True)
    trainer.ask_for_save(trainer_name)

class Trainer():

    def __init__(self, model, data_file, learning_rate=0.001, name="default_trainer"):
        self.model = model
        self.data_file = data_file
        self.tester = Tester(model=model, data_file=data_file)
        self.learning_rate = learning_rate
        self.name = name
        self.prepare_data()
        self.initialize_model()

    def initialize_model(self):
        initializer = RandomInitializer()
        initializer.initialize(self.model)
        self.tester.run()
        self.best_mse = self.tester.mse
        self.current_epoch = 0

    def prepare_data(self):
        self.tester.set_data_file(self.data_file)
        self.df = pd.read_csv(self.data_file)
        self.X = np.array([self.df["x"]])
        self.y = np.array([self.df["y"]])

    def train(self, epochs, learning_rate=None, report_interval=5e3, auto_save=True):
        self.setup_training(learning_rate=learning_rate, report_interval=report_interval)
        self.report(starting=True)
        self.plot()
        for i in range(int(epochs)):
            self.train_step(learning_rate=self.learning_rate, auto_save=auto_save)
            self.report()
        self.plot()

    def setup_training(self, learning_rate, report_interval):
        if learning_rate is not None:
            self.learning_rate = learning_rate
        self.tester.run()
        self.best_mse = self.tester.mse
        self.report_interval = report_interval

    def train_step(self, learning_rate, auto_save):
            self.model.backprop(x=self.X, y_true=self.y, learning_rate=learning_rate)
            self.tester.set_not_ready()
            self.tester.run()
            self.current_epoch += 1
            if auto_save:
                self.save_best()


    def save_best(self):
        if self.tester.mse < self.best_mse:
            self.best_mse = self.tester.mse
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
        self.report()
        self.plot()

    def report(self):
        self.tester.report()

    def plot(self):
        plt = self.tester.plot(linear=True)
        plt.show()

    def save(self, name=None):
        if name == None:
            name = self.name
        else: 
            self.name = name
        path = f'trainers/{name}.pkl'
        self.tester.run()
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