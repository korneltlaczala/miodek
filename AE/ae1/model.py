import math
import random
import time
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

from functions import BasicFunction, RastriginFunction
from evomechanisms import ChildrenGenerator, Mutator, ParentSelector, Selector


class PopulationSet():

    def __init__(self):
        self.populations = []

    def add(self, population):
        self.populations.append(population)

    def pop(self):
        return self.populations.pop()

    def __str__(self):
        return f"PopulationSet with {len(self.populations)} populations"

    def evolve(self, generations=10, verbose=True, bar_stay=False):
        iterator = (
            range(len(self.populations)) if not verbose else
            trange(len(self.populations), mininterval=0.001, ncols=120, initial=1, position=0, colour="#97cd7d")
        )
        for i in iterator:
            population = self.populations[i]
            bar_position = 1 if not bar_stay else i+1
            population.evolve(generations=generations, verbose=verbose, bar_stay=bar_stay, bar_position=bar_position, desc="")

    def plot_best_values(self, title=None, log_scale=True, plot_precision=1e-50, x_arg="generation", x_lim_right=None, x_lim_left=None, y_lim_top=None):

        plt.figure(figsize=(10, 6))
        for population in self.populations:
            x = [log[x_arg] for log in population.history.history]
            y = [log["best_value"] for log in population.history.history]
            plt.plot(x, y, label=population.label)

        def normalize_to_exponent(num):
            if num == 0:
                return 0
            exponent = math.floor(math.log10(abs(num)))
            return math.copysign(1, num) * 10**exponent
        min_val = min([population.best_value() for population in self.populations])
        min_val = normalize_to_exponent(min_val)
        if log_scale:
            plt.yscale('symlog', linthresh=max(min_val, plot_precision))

        if x_arg == "generation":
            plt.xlabel("Generation")
        if x_arg == "total_time":
            plt.xlabel("Time (s)")
        plt.ylabel("Best Fitness Value")
        plot_title = title if title is not None else f"Best Fitness Value vs. {x_arg} - {self.populations[0].fitness_function.name}"
        plt.title(plot_title)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

        top_lim = (
            y_lim_top if y_lim_top is not None else
            max(y)
        )
        left_lim = (
            x_lim_left if x_lim_left is not None else
            0
        )
        right_lim = (
            x_lim_right if x_lim_right is not None else
            max(x)
        )
        plt.ylim(bottom=0, top=top_lim)
        plt.xlim(left=left_lim, right=right_lim)
        plt.grid(True)
        plt.tight_layout()
        plt.axhline(y=10**-2, color='r', linestyle='--', label='10^-2')
        plt.show()


class PopulationHistory():
    def __init__(self, fitness_function):
        self.history = []
        self.fitness_function = fitness_function

    def log(self, generation, population, elapsed_time):
        best_vector, best_value = self.find_best(population)
        log = {
            "generation": generation,
            "unit_time": elapsed_time,
            "total_time": self.history[-1]["total_time"] + elapsed_time if len(self.history) > 0 else elapsed_time,
            "population": population,
            "best_vector": best_vector,
            "best_value": best_value,
        }
        self.history.append(log)

    def find_best(self, population):
        values = self.fitness_function.apply(population)
        return population[np.argmin(values)], min(values)

    def get_best_value_plot(self, log_scale=False):
        x = [log["generation"] for log in self.history]
        y = [log["best_value"] for log in self.history]
        if log_scale:
            plt.yscale('log')
        plt.plot(x, y)
        return plt

    def plot_best_value(self, log_scale=False):
        plt = self.get_best_value_plot(log_scale=log_scale)
        plt.show()


class Population():
    def __init__(self, size, fitness_function, interval, label=None):
        self.size = size
        self.fitness_function = fitness_function
        self.interval = interval
        self.populate(size)
        self.generation = 0
        self.history = PopulationHistory(fitness_function)
        self.history.log(self.generation, self.population, 0)
        self.label = label

    def populate(self, size):
        self.population = []
        for i in range(size):
            vector  = []
            for j in range(self.fitness_function.dim):
               vector.append(random.uniform(self.interval[0], self.interval[1]))
            self.population.append(vector)

    def evolve(self, generations=10, verbose=True, bar_stay=True, bar_position=0, desc="", parent_density=0.7, mutation_scale=0.1, elitism_rate=0.1, visualize_each_step=False):
        iterator = (
            trange(generations, desc=desc, mininterval=0.01, ncols=80, leave=bar_stay, position=bar_position, colour="#63be7b", unit="gen")
            if verbose else range(generations)
        )
        for i in iterator:
            self.generation_step(parent_density=parent_density, mutation_scale=mutation_scale, elitism_rate=elitism_rate, verbose=visualize_each_step)

    def generation_step(self, verbose=False, parent_density=0.7, mutation_scale=0.1, elitism_rate=0.1):
        self.generation += 1
        start_time = time.time()
        old_population = self.population
        children = self.generate_children(self.population, parent_density)              # krzyżowanie
        mutated_children = self.mutate(children, mutation_scale)                        # mutacja
        new_population = self.select(self.population, mutated_children, elitism_rate)   # selekcja
        self.population = new_population
        end_time = time.time()
        elapsed_time = end_time - start_time

        self.history.log(self.generation, self.population, elapsed_time)

        if verbose:
            small_size = 10
            alpha = 0.1
            samples = [old_population, children, mutated_children, new_population]
            colors = ["blue", "green", "purple", "red"]
            sizes = [50, small_size, small_size, 20]
            alphas = [1, alpha, alpha, 1]
            self.plot(samples, colors=colors, sizes=sizes, alphas=alphas)

    def generate_children(self, population, parent_density=0.7):
        parents = ParentSelector(population, self.fitness_function, parent_density).best()
        children = ChildrenGenerator(parents, self.fitness_function).pairwise()
        return children

    def mutate(self, children, mutation_scale=0.1):
        children = Mutator(children, mutation_scale=mutation_scale, permutation_chance=0.2).gaussian(children)
        return children

    def select(self, children, parents=None, elitism_rate=0.1):
        return Selector(population_size=self.size, parents=parents, children=children, fitness_function=self.fitness_function).elitism(elitism_rate=elitism_rate)

    def plot(self, populations, axis=[0, 1], colors=None, sizes=None, alphas=None):
        population_count = len(populations)
        if colors is None:
            colors = ["blue" for i in range(population_count)]
        if sizes is None:
            sizes = [20 for i in range(population_count)]
        if alphas is None:
            alphas = [1 for i in range(population_count)]
            
        for i, population in enumerate(populations):
            population = np.array(population)
            x = population[:, axis[0]]
            y = population[:, axis[1]]
            plt.scatter(x, y, color=colors[i], s=sizes[i], alpha=alphas[i])
        plt.show()
      
    def plot_population(self):
        self.plot([self.population])

    def best_value(self):
        values = self.fitness_function.apply(self.population)
        return min(values)

    def best_vector(self):
        values = self.fitness_function.apply(self.population)
        return self.population[np.argmin(values)]

    def summary(self):
        values = self.fitness_function.apply(self.population)
        best_vector = self.best_vector()
        label_len = len(self.label)
        width = 80
        border_width = int((80-label_len-6) / 2)
        print("="*width)
        print("="*border_width + self.label.center(label_len+6) + "="*border_width)
        print("="*width)
        print(f"best value: {min(values)}")
        print(f"best vector: {best_vector}")
        print()

    def evaluate(self):
        values = self.fitness_function.apply(self.population)
        return min(values)
    

if __name__ == "__main__":
    population_size = 40
    population1 = Population(population_size, BasicFunction(), [-10, 10], label="basic function")
    population2 = Population(population_size, RastriginFunction(dim=5), [-10, 10], label="rastrigin function")

    pset = PopulationSet()
    pset.add(population1)
    pset.add(population2)
    pset.evolve(generations=80, bar_stay=False)
    pset.plot_best_values(plot_precision=1e-18)