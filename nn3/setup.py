from models import *
from initializers import *
import models
import importlib
importlib.reload(models)

def train_momentum(model, epoch_counts, learning_rates, lambdas, report_interval=100, batch_size=None):
    for i in range(len(epoch_counts)):
        model.train(epochs=epoch_counts[i],
                    learning_rate=learning_rates[i],
                    batch=True,
                    optimizer="momentum",
                    momentum_lambda=lambdas[i],
                    report_interval=report_interval,
                    batch_size=batch_size)
    model.plot()

def train_rmsprop(model, epoch_counts, learning_rates, rms_beta, report_interval=100, batch_size=None):
    for i in range(len(epoch_counts)):
        model.train(epochs=epoch_counts[i],
                    learning_rate=learning_rates[i],
                    batch=True,
                    optimizer="rmsprop",
                    rms_beta=rms_beta,
                    report_interval=report_interval,
                    batch_size=batch_size)
    model.plot()