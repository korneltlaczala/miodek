import random
import matplotlib.pyplot as plt
import numpy as np

from functions import BasicFunction, RastriginFunction
from evomechanisms import ChildrenGenerator, Mutator, ParentSelector, Selector

class Visualizer():
    def __init__(self, population_dict):
        self.population_dict = population_dict

    def plot_best_values(self):
        plt.figure(figsize=(10, 6))
        for label, population in self.population_dict.items():
            x = [log["age"] for log in population.history.history]
            y = [log["best_value"] for log in population.history.history]
            plt.plot(x, y, label=label)

        plt.yscale('symlog')
        plt.xlabel("Generation")
        plt.ylabel("Best Fitness Value")
        plt.title("Best Fitness Value Over Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.axhline(y=10**-2, color='r', linestyle='--', label='10^-2')

        plt.show()


class PopulationHistory():
    def __init__(self, fitness_function):
        self.history = []
        self.fitness_function = fitness_function

    def log(self, age, population):
        best_vector, best_value = self.find_best(population)
        log = {
            "age": age,
            "population": population,
            "best_vector": best_vector,
            "best_value": best_value,
        }
        self.history.append(log)

    def find_best(self, population):
        values = self.fitness_function.apply(population)
        return population[np.argmin(values)], min(values)

    def get_best_value_plot(self, log_scale=False):
        x = [log["age"] for log in self.history]
        y = [log["best_value"] for log in self.history]
        if log_scale:
            plt.yscale('log')
        plt.plot(x, y)
        return plt

    def plot_best_value(self, log_scale=False):
        plt = self.get_best_value_plot(log_scale=log_scale)
        plt.show()



class Population():
    def __init__(self, size, fitness_function, interval):
        self.size = size
        self.fitness_function = fitness_function
        self.interval = interval
        self.populate(size)
        self.age = 0
        self.history = PopulationHistory(fitness_function)
        self.history.log(self.age, self.population)

    def populate(self, size):
        self.population = []
        for i in range(size):
            vector  = []
            for j in range(self.fitness_function.dim):
               vector.append(random.uniform(self.interval[0], self.interval[1]))
            self.population.append(vector)

    def evolve(self, generations=10, verbose=False, report_interval=10):
        for i in range(generations):
            self.generation_step(verbose)
            if i % report_interval == 0:
                print(f"generation: {i}\t\tbest value: {self.evaluate()}")

        self.summary()

    def generation_step(self, verbose):
        self.age += 1
        old_population = self.population
        children = self.generate_children(self.population)                                  # krzyżowanie
        mutated_children = self.mutate(children)                                            # mutacja
        new_population = self.select(parents=self.population, children=mutated_children)    # selekcja

        self.population = new_population

        self.history.log(self.age, self.population)

        if verbose:
            # self.show_population(self.population)
            # self.show_population(children)
            # print(f"parent count: {len(self.population)}")
            # print(f"children count: {len(children)}")
            # print(f"new population size: {len(self.population)}")
            small_size = 10
            alpha = 0.1
            samples = [old_population, children, mutated_children, new_population]
            colors = ["blue", "green", "purple", "red"]
            sizes = [50, small_size, small_size, 20]
            alphas = [1, alpha, alpha, 1]
            self.plot(samples, colors=colors, sizes=sizes, alphas=alphas)

    def generate_children(self, population, parent_density=0.7):
        selector = ParentSelector(population, self.fitness_function, parent_density)
        parents = selector.best()
        generator = ChildrenGenerator(parents, self.fitness_function)
        children = generator.pairwise()
        return children

    def mutate(self, children):
        mutator = Mutator(children, mutation_scale=0.1, permutation_chance=0.2)
        children = mutator.gaussian(children)
        return children

    def select(self, children, parents=None):
        selector = Selector(population_size=self.size, parents=parents, children=children, fitness_function=self.fitness_function)
        return selector.elitism(elitism_rate=0.1)
        # return selector.best()

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

    def show_population(self, population):
        for vector in population:
            print(vector)

    def find_best(self):
        values = self.fitness_function.apply(self.population)
        return self.population[np.argmin(values)]

    def summary(self):
        values = self.fitness_function.apply(self.population)
        best_vector = self.find_best()
        print(f"best value: {min(values)}")
        print(f"best vector: {best_vector}")

    def evaluate(self):
        values = self.fitness_function.apply(self.population)
        return min(values)

    

if __name__ == "__main__":
    population1 = Population(50, BasicFunction(), [-10, 10])
    population2 = Population(50, RastriginFunction(dim=5), [-10, 10])
    n = 100
    # population.evolve(generations=n, report_interval=1, verbose=True)
    population1.evolve(generations=n, report_interval=1000)
    population2.evolve(generations=n, report_interval=1000)
    # population.plot_population()
    # population.history.plot_best_value(log_scale=True)

    population_dict = {
        "basic function": population1,
        "rastrigin function": population2
    }
    visualizer = Visualizer(population_dict)
    visualizer.plot_best_values()