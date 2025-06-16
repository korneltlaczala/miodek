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
        layer = random.choice(child.layers)
        influenced_neuron = random.randrange(0, layer.neurons_out)
        weight_sample = other.get_weight_sample(layer.index, influenced_neuron)
        child.set_weight_sample(layer.index, influenced_neuron, weight_sample)
        layer.biases[influenced_neuron] = other.layers[layer.index].biases[influenced_neuron]
        return child

    def crossover_layer_swap(self, other):
        child = copy.deepcopy(self)
        layer1 = random.choice(child.layers)
        layer2 = other.layers[layer1.index]
        
        layer1.weights = layer2.weights.copy()
        layer1.biases = layer2.biases.copy()
        return child

    def crossover_one_weight(self, other):
        child = copy.deepcopy(self)
        layer1 = random.choice(child.layers)
        layer2 = other.layers[layer1.index]

        neuron_in_index = random.randrange(0, layer1.neurons_in)
        neuron_out_index = random.randrange(0, layer1.neurons_out)
        child.layers[layer1.index].weights[neuron_in_index, neuron_out_index] = layer2.weights[neuron_in_index, neuron_out_index]
        child.layers[layer1.index].biases[neuron_out_index] = layer2.biases[neuron_out_index]
        return child

    def mutate(self, mutation_strength=0.1):
        layer = random.choice(self.layers)
        neuron_in_index = random.randrange(0, layer.neurons_in)
        neuron_out_index = random.randrange(0, layer.neurons_out)

        layer.weights[neuron_in_index, neuron_out_index] *= (1 + random.uniform(-mutation_strength, mutation_strength))
        layer.biases[neuron_out_index] *= (1 + random.uniform(-mutation_strength, mutation_strength))

    def get_weight_sample(self, layer_index, neuron_index):
        if layer_index < 0 or layer_index >= len(self.layers):
            raise IndexError("Layer index out of bounds.")
        layer = self.layers[layer_index]
        if neuron_index < 0 or neuron_index >= layer.neurons_out:
            raise IndexError("Neuron index out of bounds.")
        return layer.weights[:, neuron_index]

    def set_weight_sample(self, layer_index, neuron_index, weight_sample):
        if layer_index < 0 or layer_index >= len(self.layers):
            raise IndexError("Layer index out of bounds.")
        layer = self.layers[layer_index]
        if neuron_index < 0 or neuron_index >= layer.neurons_out:
            raise IndexError("Neuron index out of bounds.")
        layer.weights[:, neuron_index] = copy.deepcopy(weight_sample)

    
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
            population_size=100,
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

    def train(self, target_value=0, epochs=1000, mutation_strength=0.1, verbose=True, superverbose=False):
        iterator = (
            trange(
                epochs,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
                colour="green",
                ncols=60  # Fixed width
            ) if verbose else
            range(epochs)
        )
        
        for epoch in iterator:
            if self.test_loss() <= target_value:
                print(f"Target value {target_value} reached at epoch {epoch}.")
                return self.best_model

            if verbose:
                iterator.set_postfix({"Best Loss": f"{self.test_loss():.4f}"})

            parent_pairs = self.select_parent_pairs()
            children = self.crossover(parent_pairs)
            self.mutate(children, mutation_strength=mutation_strength)
            parent_mutations = self.get_mutations(self.models, multiplier=1, mutation_strength=mutation_strength)
            children_mutations = self.get_mutations(children, multiplier=1, mutation_strength=mutation_strength)
            candidates = self.models + children + parent_mutations + children_mutations
            self.models = self.natural_selection(candidates)

            if superverbose:
                self.show_test_loss()
                self.plot()
            
    def select_parent_pairs(self, num_pairs=None, set_size=3):
        pairs = []
        if num_pairs is None:
            # num_pairs = self.population_size // 2     # Originally children count was the population size
            num_pairs = self.population_size
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
            # child1 = parent1.crossover(parent2)
            # child2 = parent2.crossover(parent1)
            # child1 = parent1.crossover_layer_swap(parent2)
            # child2 = parent2.crossover_layer_swap(parent1)
            child1 = parent1.crossover_one_weight(parent2)
            child2 = parent2.crossover_one_weight(parent1)
            children.append(child1)
            children.append(child2)
        return children

    def mutate(self, children, mutation_probability=0.1, mutation_strength=0.1):
        for child in children:
            if random.random() < mutation_probability:
                child.mutate(mutation_strength=mutation_strength)

    def get_mutations(self, sample, multiplier=5, mutation_strength=0.1):
        mutations = []
        for model in sample:
            for i in range(multiplier):
                mutated_model = copy.deepcopy(model)
                mutated_model.mutate(mutation_strength=mutation_strength)
                mutations.append(mutated_model)
        return mutations

    def natural_selection(self, candidates):
        min_loss = min(model.test_loss() for model in candidates)
        max_loss = max(model.test_loss() for model in candidates)
        spread = max_loss - min_loss
        # Elitism: keep the best models directly
        elite_count = max(1, self.population_size // 10)
        elites = sorted(candidates, key=lambda m: m.test_loss())[:elite_count]
        # Select the rest based on a weighted probability
        non_elites = [m for m in candidates if m not in elites]
        weights = [max_loss - model.test_loss() + 1e-8 for model in non_elites]  # avoid zero
        total = sum(weights)
        probabilities = [w / total for w in weights]
        chosen_candidates = choice(
            non_elites,
            size=self.population_size - elite_count,
            p=probabilities,
            replace=False
        )
        # Combine elites and selected
        chosen_candidates = list(elites) + list(chosen_candidates)
        return chosen_candidates
            
    def get_population(self):
        return self.population

    def get_population_size(self):
        return self.population_size

    def plot_dataset(self):
        self.models[0].data.plot_test()

    def plot(self):
        self.best_model.plot()

    def train_loss(self):
        return self.best_model_train.train_loss()

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

    def classification_performance_summary(self):
        self.best_model.classification_performance_summary()

    @property
    def best_model_train(self):
        best_model = min(self.models, key=lambda model: model.train_loss())
        return best_model

    @property
    def best_model(self):
        best_model = min(self.models, key=lambda model: model.test_loss())
        return best_model

    def __getitem__(self, index):
        return self.models[index]