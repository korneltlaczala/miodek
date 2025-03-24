from models import *
from initializers import *

def main():
    # dataset_name = "data"
    # arch = MLPArchitecture(3, [10, 10], 1)
    dataset_name = "square-simple"
    # arch = MLPArchitecture(1, [10, 10], 1)
    # arch = MLPArchitecture(1, [20, 50, 20], 1)
    arch = MLPArchitecture(1, [50, 100, 50], 1)
    model = MLP(architecture=arch, dataset_name=dataset_name)
    model.evaluate()
    model.plot()
    model.train(epochs=2000, learning_rate=0.1, batch=True)
    model.plot()
    model.train(epochs=2000, learning_rate=0.1, batch=True)
    model.plot()
    model.train(epochs=2000, learning_rate=0.01, batch=True)
    model.plot()


if __name__ == "__main__":
    main()