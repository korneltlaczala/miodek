import random
import matplotlib.pyplot as plt

from functions import BasicFunction
from evomechanisms import ChildrenGenerator, ParentSelector

class PopulationHistory():
    pass

class Population():
    def __init__(self, size, fitness_function, interval):
        self.size = size
        self.fitness_function = fitness_function
        self.interval = interval
        self.populate(size)
        self.age = 0

    def populate(self, size):
        self.population = []
        for i in range(size):
            vector  = []
            for j in range(self.fitness_function.dim):
               vector.append(random.uniform(self.interval[0], self.interval[1]))
            self.population.append(vector)

    def evolve(self):
        self.age += 1

        # krzyżowanie
        children = self.generate_children(self.population)

        # mutacja
        mutated_children = self.mutate(children)

        # selekcja
        # nowa populacja

        # self.show_population(self.population)
        # self.show_population(children)
        print(f"parent count: {len(self.population)}")
        print(f"children count: {len(children)}")
        self.plot([self.population, children], colors=["blue", "red"], sizes=[100, 20], alphas=[1, 0.5])

    def generate_children(self, population, parent_density=0.7):
        selector = ParentSelector(population, self.fitness_function, parent_density)
        parents = selector.best()
        generator = ChildrenGenerator(parents, self.fitness_function)
        children = generator.pairwise()
        return children

    def mutate(self, children):
        return children

    def plot(self, populations, axis=[0, 1], colors=None, sizes=None, alphas=None):
        population_count = len(populations)
        if colors is None:
            pass
        if sizes is None:
            sizes = [1 for i in range(population_count)]
        if alphas is None:
            alphas = [1 for i in range(population_count)]
            
        for i, population in enumerate(populations):
            size = len(population)
            x = [population[i][axis[0]] for i in range(size)]
            y = [population[i][axis[1]] for i in range(size)]
            plt.scatter(x, y, color=colors[i], s=sizes[i], alpha=alphas[i])
        plt.show()
      
    def plot_population(self):
        self.plot(self.population)

    def show_population(self, population):
        for vector in population:
            print(vector)
    

if __name__ == "__main__":
    population = Population(10, BasicFunction(), [0, 10])
    # population.plot_population()
    population.evolve()