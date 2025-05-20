import random
import matplotlib.pyplot as plt

from functions import BasicFunction

class PopulationHistory():
    pass

class Population():
    def __init__(self, size, function, interval):
        self.size = size
        self.function = function
        self.interval = interval
        self.populate(size)
        self.age = 0

    def populate(self, size):
        self.population = []
        for i in range(size):
            vector  = []
            for j in range(self.function.dim):
               vector.append(random.uniform(self.interval[0], self.interval[1]))
            self.population.append(vector)

    def evolve(self):
        self.age += 1

        # krzyżowanie
        new_population = self.generate_children(self.population)
        

        # mutacja
        # selekcja
        # nowa populacja

    def generate_children(self, population):
        fitness_values = self.function.apply(population)
        print(fitness_values)
        return

    def plot(self, axis=[0, 1]):
        x = [self.population[i][axis[0]] for i in range(self.size)]
        y = [self.population[i][axis[1]] for i in range(self.size)]
        plt.scatter(x, y)
        plt.show()
      

if __name__ == "__main__":
    population = Population(10, BasicFunction(), [0, 10])
    population.plot()
    population.evolve()