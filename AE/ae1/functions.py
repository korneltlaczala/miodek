from abc import ABC, abstractmethod
import math

class Function(ABC):

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def calculate(self, vector):
        pass

    def apply(self, vector_list):
        return [self.calculate(vector) for vector in vector_list]



class BasicFunction(Function):

    def __init__(self):
        self.dim = 3

    @property
    def name(self):
        return f"Basic Function"
    
    def calculate(self, vector):
        x = vector[0]
        y = vector[1]
        z = vector[2]
        return x*x + y*y + 2*z*z

class RastriginFunction(Function):

    def __init__(self, dim):
        self.A = 10
        self.dim = dim

    @property
    def name(self):
        # return f"Rastrigin Function ({self.dim}D)"
        return f"Rastrigin Function"
    
    def calculate(self, vector):
        value = self.A * self.dim
        for i in range(self.dim):
            value += vector[i] * vector[i] - self.A*math.cos(2*math.pi*vector[i])
        return value

        