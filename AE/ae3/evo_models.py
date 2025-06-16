import random
import copy
from numpy.random import choice
from models import *
from tqdm import tqdm, trange

class EvoMLP(MLP):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def crossover(self, other):
        child = copy.deepcopy(self)
        print("Performing crossover between models...")
        # Implement crossover logic here
        return child

    def mutate(self):
        # Implement mutation logic here
        pass

class ModelPopulation():
    def __init__(
            self,
            architecture,
            dataset_name,
            data_dir="../data",
            loss_function="mse",
            initializer=None,
            activation_func=None,
            last_layer_activation_func=None,
            target_precision=1e-2,

            model_class=EvoMLP,
            population_size=10,
            population_name="model_population",
            ):

        self.architecture = architecture
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.loss_function = loss_function
        self.initializer = initializer
        self.activation_func = activation_func
        self.last_layer_activation_func = last_layer_activation_func
        self.target_precision = target_precision

        self.model_class = model_class
        self.population_size = population_size
        self.population_name = population_name
        self.generate_population()
        
    def generate_population(self):
        self.models = [
            self.model_class(
                architecture=self.architecture,
                dataset_name=self.dataset_name,
                data_dir=self.data_dir,
                loss_function=self.loss_function,
                initializer=self.initializer,
                activation_func=self.activation_func,
                last_layer_activation_func=self.last_layer_activation_func,
                target_precision=self.target_precision
            )
            for _ in range(self.population_size)
        ]

    def train(self, target_value=0, epochs=1000, verbose=False):
        iterator = (
            trange(
                epochs,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                colour="lightblue",
                ncols=60  # Fixed width
            ) if verbose else
            range(epochs)
        )
        
        for epoch in iterator:
            if self.test_loss() <= target_value:
                print(f"Target value {target_value} reached at epoch {epoch}.")
                return self.best_model

            parent_pairs = self.select_parent_pairs()
            children = self.crossover(parent_pairs)
            self.mutate(children)
            candidates = self.models + children
            self.models = self.natural_selection(candidates)
            
    def select_parent_pairs(self, num_pairs=None, set_size=3):
        pairs = []
        if num_pairs is None:
            num_pairs = self.population_size // 2
        for _ in range(num_pairs):
            parent1 = self.tournament_selection(random.sample(self.models, set_size))
            parent2 = self.tournament_selection(random.sample(self.models, set_size))
            pair = (parent1, parent2)
            pairs.append(pair)
        return pairs

    def tournament_selection(self, candidates):
        return max(candidates, key=lambda model: model.test_loss())

    def crossover(self, parent_pairs):
        children = []
        for parent1, parent2 in parent_pairs:
            child1 = parent1.crossover(parent2)
            child2 = parent2.crossover(parent1)
            children.append(child1)
            children.append(child2)
        return children

    def mutate(self, children, mutation_probability=0.1):
        for child in children:
            if random.random() < mutation_probability:
                child.mutate()

    def natural_selection(self, candidates):
        min_loss = min(model.test_loss() for model in candidates)
        max_loss = max(model.test_loss() for model in candidates)
        spread = max_loss - min_loss
        chosen_candidates = choice(
            candidates,
            size=self.population_size,
            p=[(model.test_loss() - min_loss + spread/2) / (spread + 1e-8) for model in candidates],
            replace=False
        )
        return chosen_candidates.tolist()
            
    def get_population(self):
        return self.population

    def get_population_size(self):
        return self.population_size

    def plot_dataset(self):
        self.models[0].data.plot_test()

    def plot(self):
        self.best_model.plot()

    def test_loss(self):
        return self.best_model.test_loss()

    def show_test_loss(self):
        print(f"Best model test loss: {self.test_loss()}")

    def plot_at_index(self, index=0):
        if index < 0 or index >= self.population_size:
            raise IndexError("Index out of bounds for population.")
        self.models[index].plot()

    def evaluate_at_index(self, index=0):
        if index < 0 or index >= self.population_size:
            raise IndexError("Index out of bounds for population.")
        return self.models[index].evaluate()

    @property
    def best_model(self):
        best_model = min(self.models, key=lambda model: model.test_loss())
        return best_model

    def __getitem__(self, index):
        return self.models[index]