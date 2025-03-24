from models import *
from initializers import *

def main():
    dataset_name = "square-large"
    # arch = MLPArchitecture(1, [50, 100, 50], 1)
    arch = MLPArchitecture(1, [1000], 1)

    batch = True
    report_interval = 10
    
    lambdas = [0.01, 0.1, 0.3, 0.5, 0.8, 0.8, 0.8, 0.9, 0.95]
    epoch_counts = [100] * 9
    learning_rates = [0.1] * 4 + [0.01] * 5

    model = MLP(architecture=arch, dataset_name=dataset_name, initializer=XavierNormalInitializer())
    model.evaluate()
    
    for i in range(9):
        model.plot()
        model.train(epochs=epoch_counts[i], learning_rate=learning_rates[i], batch=True, momentum_lambda=lambdas[i], report_interval=report_interval, batch_size=1)
    # model.plot()
    # model.train(epochs=100, learning_rate=0.1, batch=batch, momentum_lambda=0.01, report_interval=report_interval)
    # model.plot()
    # model.train(epochs=100, learning_rate=0.1, batch=batch, momentum_lambda=0.1, report_interval=report_interval)
    # model.plot()
    # model.train(epochs=100, learning_rate=0.1, batch=batch, momentum_lambda=0.5, report_interval=report_interval)
    # model.plot()
    # model.train(epochs=2000, learning_rate=0.05, batch=batch, momentum_lambda=0.5, report_interval=report_interval)
    # model.plot()
    # model.train(epochs=2000, learning_rate=0.05, batch=batch, momentum_lambda=0.9, report_interval=report_interval)
    # model.plot()
    # model.train(epochs=2000, learning_rate=0.01, batch=batch, momentum_lambda=0.9, report_interval=report_interval)
    model.plot()


if __name__ == "__main__":
    main()