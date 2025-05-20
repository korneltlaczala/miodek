import random
import numpy as np


class ChildrenGenerator:

    def __init__(self, parents, fitness_function):
        self.parents = parents
        self.fitness_function = fitness_function

    def pairwise(self):
        children = []
        for i in range(len(self.parents)):
            for j in range(i+1, len(self.parents)):
                child1, child2 = self.one_point_crossover(self.parents[i], self.parents[j])
                children.append(child1)
                children.append(child2)
                # children.append(self.one_point_crossover(self.parents[i], self.parents[j]))
        return children

    def one_point_crossover(self, parent1, parent2):
        dim = len(parent1)
        k = random.randint(0, dim-1)
        child1 = np.concatenate((parent1[:k], parent2[k:]))
        child2 = np.concatenate((parent2[:k], parent1[k:]))
        return child1, child2


class ParentSelector:

    def __init__(self, population, fitness_function, parent_density):
        self.population = np.copy(population)
        self.fitness_function = fitness_function
        self.parent_density = parent_density

    def best(self):
        size = len(self.population)
        parent_count = int(self.parent_density * size)
        values = self.fitness_function.apply(self.population)
        self.population = self.population[np.argsort(values)]
        return self.population[size-parent_count:]

    def all(self):
        return self.population

    