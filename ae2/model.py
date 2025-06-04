import os
import random
import time

from tqdm import tqdm
from matplotlib import pyplot as plt



class Shape:
    def __init__(self, width, height, value):
        self.width = width
        self.height = height
        self.value = value

    def __str__(self):
        return f'width: {self.width}, height: {self.height}, value: {self.value}'

class Rectangle:
    def __init__(self, x, y, shape):
        self.x = x
        self.y = y
        self.shape = shape

    def fits_in_the_circle(self, radius):
        for corner in self.corners:
            if corner[0]*corner[0] + corner[1]*corner[1] > radius*radius:
                return False
        return True

    def touches_the_circle(self, radius):
        for corner in self.corners:
            if corner[0]*corner[0] + corner[1]*corner[1] < radius*radius:
                return True
        return False

    @property
    def corners(self):
        return [(self.x, self.y),
                (self.x + self.shape.width, self.y),
                (self.x, self.y + self.shape.height),
                (self.x + self.shape.width, self.y + self.shape.height)]

    @property
    def center(self):
        return (self.x + self.shape.width/2, self.y + self.shape.height/2)

class Individual:
    def __init__(self, radius):
        self.radius = radius
        self.columns = []

    def match_width(self, shapes):
        self.strip()
        self.random_fill(shapes)

    def strip(self):
        curr_width = self.width
        while curr_width > self.radius*2:
            column = self.columns.pop()
            curr_width -= column.width
    
    def random_fill(self, _shapes):
        shapes = _shapes.copy()
        max_width = self.radius*2
        while self.width < max_width and len(shapes) > 0:
            shape = random.choice(shapes)
            if shape.width + self.width > max_width:
                shapes.remove(shape)
                continue
            self.columns.append(shape)

    def crossover(self, other, shapes):
        child = Individual(self.radius)
        # child.columns = random.sample(self.columns, len(self.columns)//2) + random.sample(other.columns, len(other.columns)//2)
        crossover_point = random.randint(0, len(self.columns)-1)
        child.columns = self.columns[:crossover_point] + other.columns[crossover_point:]
        child.match_width(shapes)
        return child

    def mutate(self, mutation_rate):
        if random.random() < mutation_rate:
            random.shuffle(self.columns)
        # random.shuffle(self.columns)

    def generate_rectangles(self):
        rectangles = []
        x = -self.radius
        for shape in self.columns:
            y = -self.radius
            while y + shape.height <= self.radius:
                rectangles.append(Rectangle(x, y, shape))
                y += shape.height
            x += shape.width
        return rectangles

    def evaluate(self):
        value = 0
        rectangles = self.generate_rectangles()
        for rectangle in rectangles:
            if rectangle.fits_in_the_circle(self.radius):
                value += rectangle.shape.value
        return value

    def plot(self, hide_outsiders=False):
        fig, ax = plt.subplots(figsize=(8, 8))
        circle = plt.Circle((0, 0), self.radius, edgecolor='black', facecolor='none')
        ax.add_artist(circle)
        for rectangle in self.generate_rectangles():
            if not rectangle.touches_the_circle(self.radius) and hide_outsiders:
                continue
            rect = plt.Rectangle(
                (rectangle.x, rectangle.y),
                rectangle.shape.width,
                rectangle.shape.height,
                edgecolor='black',
                facecolor='green' if rectangle.fits_in_the_circle(self.radius) else 'none',
                alpha=0.5
            )
            ax.add_artist(rect)
            ax.text(rectangle.center[0], rectangle.center[1], rectangle.shape.value, ha='center', va='center', fontsize=8)
        ax.set_aspect('equal')
        ax.set_xlim(-self.radius-1, self.radius+1)
        ax.set_ylim(-self.radius-1, self.radius+1)
        ax.set_axis_off()
        plt.title(f"Total value: {self.evaluate()}")
        plt.show()

    @property
    def width(self):
        return sum([shape.width for shape in self.columns])

    def __str__(self):
        return f'width: {self.width}, columns: {[shape.width for shape in self.columns]}'


class Population:
    def __init__(self, size, shapes):
        self.size = size
        self.shapes = shapes
        self.MUTATION_RATE = 0.2
        self.individuals = []

    def next_generation(self):
        parents = self.get_parents()
        children = self.get_children(parents)
        new_population = Population(self.size, self.shapes)
        new_population.individuals = children
        new_population.add_individual(self.best_individual())
        new_population.remove_individual(new_population.worst_individual())
        return new_population
        
    def get_parents(self):
        parents = []
        for i in range(self.size):
            sample_size = max(2, self.size * 0.1)
            parent1 = self.tournament_selection(random.sample(self.individuals, sample_size))
            parent2 = self.tournament_selection(random.sample(self.individuals, sample_size))
            parents.append((parent1, parent2))
        return parents

    def get_children(self, parents):
        children = []
        for parent1, parent2 in parents:
            child = parent1.crossover(parent2, self.shapes)
            child.mutate(self.MUTATION_RATE)
            children.append(child)
        return children

    def tournament_selection(self, sample):
        return max(sample, key=lambda individual: individual.evaluate())

    def add_individual(self, individual):
        self.individuals.append(individual)

    def remove_individual(self, individual):
        self.individuals.remove(individual)

    def best_individual(self):
        return max(self.individuals, key=lambda individual: individual.evaluate())

    def worst_individual(self):
        return min(self.individuals, key=lambda individual: individual.evaluate())

    @property
    def best_evaluation(self):
        return max([individual.evaluate() for individual in self.individuals])

    def __str__(self):
        return f"Population with {len(self.individuals)} individuals"

class Cutting:

    def __init__(self, radius, population_size):
        self.radius = radius
        self.population_size = population_size
        self.data_folder = 'data'
        self.data_file = os.path.join(self.data_folder, f'r{radius}.csv')
        self.read_shapes()
        self.populate()
        self.history = []


    def read_shapes(self):
        self.shapes = []
        with open(self.data_file, 'r') as file:
            shapes = [line.strip().split(',') for line in file]
        for shape in shapes:
            self.shapes.append(Shape(
                width=int(shape[0]),
                height=int(shape[1]),
                value=int(shape[2])
            ))
        
    def populate(self):
        self.population = Population(self.population_size, self.shapes)
        for i in range(self.population_size):
            individual = Individual(self.radius)
            individual.random_fill(self.shapes)
            self.population.add_individual(individual)
        
    def train(self, target_value=None, iterations=1000, verbose=True):
        iterator = (
            tqdm(range(iterations), bar_format="{l_bar}%s{bar}%s{r_bar}" % ("\033[94m", "\033[0m"), desc=f"Best Eval: {self.population.best_evaluation}")  if verbose else
            range(iterations)
        )
        for i in iterator:
            if target_value is not None and self.population.best_evaluation >= target_value:
                if verbose:
                    iterator.set_description(f"Best Eval: {self.population.best_evaluation}")
                break
            if verbose:
                iterator.set_description(f"Best Eval: {self.population.best_evaluation}")

            self.next_generation()

        print(f"===============================")
        print(f"Finished training after {i+1} iterations")
        print(f"Best evaluation: {self.population.best_evaluation}")
        print(f"===============================")

    def next_generation(self):
        self.history.append(self.population.best_evaluation)
        self.population = self.population.next_generation()

    def print_shapes(self):
        for shape in self.shapes:
            print(shape)

    def plot_history(self):

        best_val = self.history[-1]
        first_best_gen = self.history.index(best_val)
        relevant_history = self.history[:int(first_best_gen*1.2)+1]

        plt.figure(figsize=(7, 5))
        plt.plot(relevant_history)
        plt.grid(True)
        plt.title("Best evaluation for each generation")
        plt.xlabel("Generations")
        plt.ylabel("Best evaluation")
        plt.show()

    def plot_best(self):
        self.population.best_individual().plot(hide_outsiders=True)

    def plot_result(self):
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        axes[0].plot(self.history)
        axes[1].axis('off')
        axes[1].imshow(self.population.best_individual().plot(hide_outsiders=True))
        plt.show()

if __name__ == '__main__':
    radius = 800
    target = 30000
    cutting = Cutting(radius=radius, population_size=10)
    # cutting.train(target_value=target, iterations=100, verbose=True)
    cutting.train(iterations=100, verbose=True)