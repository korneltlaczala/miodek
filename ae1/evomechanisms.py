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

class Selector:

    def __init__(self, population_size, parents, children, fitness_function):
        self.population_size = population_size
        self.parents = np.array(parents)
        self.children = np.array(children)
        self.fitness_function = fitness_function

    def best(self):
        candidates = np.concatenate((self.parents, self.children), axis=0)
        values = self.fitness_function.apply(candidates)
        return candidates[np.argsort(values)][:self.population_size]

    def elitism(self, elitism_rate):
        safe_parent_count = int(elitism_rate * self.population_size)
        values = self.fitness_function.apply(self.parents)
        safe_parents = self.parents[np.argsort(values)][:safe_parent_count]

        candidates = np.concatenate((self.parents[np.argsort(values)][safe_parent_count:], self.children), axis=0)
        values = self.fitness_function.apply(candidates)
        chosen_candidates = candidates[np.argsort(values)][:self.population_size-safe_parent_count]

        return np.concatenate((safe_parents, chosen_candidates), axis=0)


    
class Mutator:

    def __init__(self, population, mutation_scale, permutation_chance=0.2):
        self.population = np.array(population)
        self.mutation_scale = mutation_scale
        self.permutation_chance = permutation_chance
        self.calculate_ranges()

    def calculate_ranges(self):
        self.ranges = self.population.max(axis=0) - self.population.min(axis=0)

    def gaussian(self, original_sample):
        sample = np.copy(original_sample)
        for vector in sample:
            for i in range(len(vector)):
                    vector[i] += random.gauss(0, self.ranges[i] * self.mutation_scale)
        sample = self.permutate(sample)
        return sample

    def uniform(self, sample):
        return sample

    def permutate(self, sample):
        for vector in sample:
            if random.random() < self.permutation_chance:
                vector = vector[np.random.permutation(len(vector))]
        return sample