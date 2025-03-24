from models import *
from initializers import *

def main():
    dataset_name = "square-large"
    arch = MLPArchitecture(1, [50, 100, 50], 1)
    # arch = MLPArchitecture(1, [1000], 1)

    batch = True
    momentum_lambda = 0.9
    report_interval = 10
    model = MLP(architecture=arch, dataset_name=dataset_name, initializer=XavierNormalInitializer())
    model.evaluate()
    model.plot()
    model.train(epochs=1000, learning_rate=0.1, batch=batch, momentum_lambda=0.01, report_interval=report_interval)
    model.plot()
    model.train(epochs=1000, learning_rate=0.1, batch=batch, momentum_lambda=0.1, report_interval=report_interval)
    model.plot()
    model.train(epochs=1000, learning_rate=0.1, batch=batch, momentum_lambda=0.5, report_interval=report_interval)
    model.plot()
    model.train(epochs=2000, learning_rate=0.05, batch=batch, momentum_lambda=0.5, report_interval=report_interval)
    model.plot()
    model.train(epochs=2000, learning_rate=0.05, batch=batch, momentum_lambda=0.9, report_interval=report_interval)
    model.plot()
    model.train(epochs=2000, learning_rate=0.01, batch=batch, momentum_lambda=0.9, report_interval=report_interval)
    model.plot()


if __name__ == "__main__":
    main()