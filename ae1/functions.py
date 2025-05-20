import math

class Function():

    def apply(self, vector_list):
        return [self.calculate(vector) for vector in vector_list]


class BasicFunction(Function):

    def __init__(self):
        self.dim = 3
    
    def calculate(self, vector):
        x = vector[0]
        y = vector[1]
        z = vector[2]
        return x*x + y*y + 2*z*z

class RastringerFunction(Function):

    def __init__(self, dim):
        self.A = 10
        self.dim = dim
    
    def calculate(self, vector):
        value = self.A * self.dim
        for i in range(self.dim):
            value += vector[i] * vector[i] - self.A*math.cos(2*math.pi*vector[i])
        return value

        