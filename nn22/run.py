from models import *

def main():
    # dataset_name = "data"
    # arch = MLPArchitecture(3, [10, 10], 1)
    dataset_name = "square-simple"
    arch = MLPArchitecture(1, [10, 10], 1)
    model = MLP(architecture=arch, dataset_name=dataset_name)
    initializer = NormalInitializer()
    initializer.initialize(model)
    model.evaluate()
    model.plot()
    model.train(epochs=10000, learning_rate=0.01)
    model.plot()

if __name__ == "__main__":
    main()