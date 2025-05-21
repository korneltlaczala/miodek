import random
import matplotlib.pyplot as plt
import numpy as np

from functions import BasicFunction, RastriginFunction
from evomechanisms import ChildrenGenerator, Mutator, ParentSelector, Selector

class Population():
    def __init__(self, size, fitness_function, interval):
        self.size = size
        self.fitness_function = fitness_function
        self.interval = interval
        self.populate(size)
        self.age = 0
        self.history = []

    def populate(self, size):
        self.population = []
        for i in range(size):
            vector  = []
            for j in range(self.fitness_function.dim):
               vector.append(random.uniform(self.interval[0], self.interval[1]))
            self.population.append(vector)

    def evolve(self, generations=10, verbose=False, report_interval=10):
        for i in range(generations):
            self.pass_generation(verbose)
            if i % report_interval == 0:
                print(f"generation: {i}\t\tbest value: {self.evaluate()}")

        self.summary()

    def pass_generation(self, verbose):
        self.age += 1
        old_population = self.population
        children = self.generate_children(self.population)                                  # krzyżowanie
        mutated_children = self.mutate(children)                                            # mutacja
        new_population = self.select(parents=self.population, children=mutated_children)    # selekcja

        self.population = new_population

        if verbose:
            # self.show_population(self.population)
            # self.show_population(children)
            # print(f"parent count: {len(self.population)}")
            # print(f"children count: {len(children)}")
            # print(f"new population size: {len(self.population)}")
            small_size = 10
            alpha = 0.0
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
        return selector.elitism(elitism_rate=0.2)
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

    def find_min(self):
        values = self.fitness_function.apply(self.population)
        return self.population[np.argmin(values)]

    def summary(self):
        values = self.fitness_function.apply(self.population)
        v_min = self.find_min()
        print(f"best value: {min(values)}")
        print(f"best vector: {v_min}")

    def evaluate(self):
        values = self.fitness_function.apply(self.population)
        return min(values)

    

if __name__ == "__main__":
    population = Population(30, BasicFunction(), [-10, 10])
    # population = Population(50, RastriginFunction(dim=5), [-10, 10])
    n = 600
    # population.evolve(generations=n, report_interval=1, verbose=True)
    population.evolve(generations=n, report_interval=10)
    population.plot_population()