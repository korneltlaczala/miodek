import os
import random

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

    def random_fill(self, shapes):
        max_width = self.radius*2
        while self.width < max_width and len(shapes) > 0:
            shape = random.choice(shapes)
            if shape.width + self.width > max_width:
                shapes.remove(shape)
                continue
            self.columns.append(shape)

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
        fig, ax = plt.subplots()
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
        plt.show()

    @property
    def width(self):
        return sum([shape.width for shape in self.columns])

    def __str__(self):
        return f'width: {self.width}, columns: {[shape.width for shape in self.columns]}'


class Population:
    def __init__(self):
        self.individuals = []

    def add_individual(self, individual):
        self.individuals.append(individual)


class Cutting:

    def __init__(self, radius, population_size):
        self.radius = radius
        self.population_size = population_size
        self.data_folder = 'data'
        self.data_file = os.path.join(self.data_folder, f'r{radius}.csv')
        self.read_shapes()
        self.populate()


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
        population = Population()
        for i in range(self.population_size):
            individual = Individual(self.radius)
            individual.random_fill(self.shapes)
            population.add_individual(individual)
            print(individual.evaluate())
            individual.plot()
        

    def print_shapes(self):
        for shape in self.shapes:
            print(shape)


if __name__ == '__main__':
    Cutting(radius=800, population_size=1)
