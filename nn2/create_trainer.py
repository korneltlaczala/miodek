from models import *
from training import Trainer
import numpy as np
import pandas as pd


data_file = "data/square-simple-training.csv"
trainer_name = "train_trainer"
# trainer_name = "trainer_app"
# trainer_name = "trainer_square"
model = MLP(1, [16, 30, 16], 1)
trainer = Trainer(model, data_file, name=trainer_name)
trainer.save()

