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

    def evolve(self, generations=10, verbose=False, report_count=1):
        report_interval = int(generations / report_count)
        for i in range(generations):
            self.pass_generation(verbose)
            if i % report_interval == 0:
                print(f"generation: {i}:\t", end="")
                self.evaluate()

        self.evaluate()

    def pass_generation(self, verbose):
        self.age += 1
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
            alpha = 0.1
            samples = [self.population, children, mutated_children, new_population]
            colors = ["blue", "red", "purple", "green"]
            sizes = [30, small_size, small_size, small_size]
            alphas = [1, alpha, alpha, 0.7]
            self.plot(samples, colors=colors, sizes=sizes, alphas=alphas)

    def generate_children(self, population, parent_density=0.7):
        selector = ParentSelector(population, self.fitness_function, parent_density)
        parents = selector.best()
        generator = ChildrenGenerator(parents, self.fitness_function)
        children = generator.pairwise()
        return children

    def mutate(self, children):
        mutator = Mutator(children, 0.1)
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

    def evaluate(self):
        values = self.fitness_function.apply(self.population)
        print(f"best value: {min(values)}")
    

if __name__ == "__main__":
    # population = Population(200, BasicFunction(), [-10, 10])
    population = Population(200, RastriginFunction(dim=5), [-10, 10])
    n = 100
    # population.evolve(generations=n, report_count=n, verbose=True)
    population.evolve(generations=n, report_count=n)
    population.plot_population()