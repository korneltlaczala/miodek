from abc import abstractmethod

import numpy as np
from models import *

def run():
    data_file = "data/square-simple-test.csv"
    # trainer_name = "trainer_square_2000"
    trainer_name = "trainer_square"
    # trainer_name = "trainer_app"
    # data_file = "data/multimodal-large-test.csv"
    # trainer_name = "trainer_multimodal"
    # model = MLP(1, [16, 30, 16], 1)
    # trainer = Trainer(model, data_file, name=trainer_name)
    trainer = Trainer.load(trainer_name)
    trainer.test()
    trainer.train(epochs=1e4, learning_rate=0.1, auto_save=True)
    trainer.ask_for_save(trainer_name)

class Trainer():

    def __init__(self, model, data_file, learning_rate=0.001, report_interval=1e3, name="default_trainer"):
        self.data_file = data_file
        self.tester = Tester(data_file=data_file)
        self.set_model(model)
        self.learning_rate = learning_rate
        self.report_interval = report_interval
        self.name = name
        self.prepare_data()
        self.initialize_model()

    def initialize_model(self):
        initializer = RandomInitializer()
        initializer.initialize(self.model)
        self.tester.run()
        self.current_epoch = 0
        self.best_mse = float('inf')
        self.update_best_model()
        self.update_init_model()
        
    def prepare_data(self):
        self.tester.set_data_file(self.data_file)
        self.df = pd.read_csv(self.data_file)
        self.X = np.array([self.df["x"]])
        self.y = np.array([self.df["y"]])

    def set_model(self, model):
        self.model = model
        self.tester.set_model(model)
        self.evaluate_model()

    def evaluate_model(self):
        self.tester.set_not_ready()
        self.tester.run()

    def train(self, epochs, learning_rate=None, report_interval=5e3, auto_save=True):
        self.setup_training(learning_rate=learning_rate, report_interval=report_interval)
        self.report(starting=True)
        self.plot()
        for i in range(int(epochs)):
            self.train_step(learning_rate=self.learning_rate)
            self.report()
        self.plot()
        if auto_save:
            self.save_best()

    def train_in_app(self, epochs, learning_rate=None, report_interval=5e3, auto_save=False):
        self.setup_training(learning_rate=learning_rate, report_interval=report_interval)
        for i in range(int(epochs)):
            self.train_step(learning_rate=self.learning_rate)
        if auto_save:
            self.save_best()

    def setup_training(self, learning_rate, report_interval):
        if learning_rate is not None:
            self.learning_rate = learning_rate
        self.report_interval = report_interval
        self.update_init_model()

    def train_step(self, learning_rate):
            self.model.backprop(x=self.X, y_true=self.y, learning_rate=learning_rate)
            self.evaluate_model()
            self.current_epoch += 1
            self.update_best_model()

    def update_best_model(self):
        if self.mse < self.best_mse:
            self.best_mse = self.mse
            self.best_model = self.model.get_copy()
            self.best_model_age = self.current_epoch
            print(f"new best, epoch: {self.current_epoch}\t\tmse: {round(self.mse, 2)}")

    def update_init_model(self):
        self.init_mse = self.mse
        self.init_model = self.model.get_copy()
        self.init_model_age = self.current_epoch
        self.best_mse_before_init = self.best_mse
        self.best_model_before_init = self.best_model.get_copy()
        self.best_model_before_init_age = self.best_model_age


    def report(self, starting=False):
        if starting:
            self.report_start()
        if self.current_epoch % self.report_interval == 0:
            print(f"epoch: {self.current_epoch}\t\tmse: {round(self.mse, 2)}")

    def report_start(self):
        print("----------------------")
        print(f"Starting training")
        print(f"model age: {self.current_epoch} epochs")
        print(f"starting mse: {self.mse}")
        print("----------------------")

    def test(self):
        self.report()
        self.plot()

    def plot(self):
        plt = self.tester.plot(linear=True)
        plt.show()

    def get_fig(self, model):
        tester = Tester(model=model, data_file=self.data_file)
        return tester.get_fig(linear=True)

    def undo_training(self):
        self.set_model(self.init_model)
        self.current_epoch = self.init_model_age
        self.best_mse = self.best_mse_before_init
        self.best_model = self.best_model_before_init.get_copy()
        self.best_model_age = self.best_model_before_init_age

    def save_with_best_model(self, name=None):
        self.set_model(self.best_model)
        self.current_epoch = self.best_model_age
        self.save(name)

    def save_best(self):
        saved_trainer = self.get_copy()
        saved_trainer.save_with_best_model()

    def save(self, name=None):
        if name == None:
            name = self.name
        else: 
            self.name = name
        path = f'trainers/{name}.pkl'
        self.tester.run()
        self.update_init_model()
        with open(path, 'wb') as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)
        print(f"Trainer (age: {self.current_epoch}) saved as {name}.\t mse: {round(self.mse, 2)}")

    def ask_for_save(self, name):
        response = None
        while response not in ['y', 'n']:
            response = input("Save trainer? (y/n) ")
        if response == 'y':
            self.save(name)
        else:
            print("Trainer not saved")

    def get_copy(self):
        return copy.deepcopy(self)
    
    @classmethod
    def load(self, name):
        path = f'trainers/{name}.pkl'
        with open(path, 'rb') as file:
            return pickle.load(file)

    @property
    def mse(self):
        return self.tester.get_mse()

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