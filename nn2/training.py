from models import MLP

class Trainer():

    def __init__(self, model):
        self.model = model

        self.initialize_model()

    def initialize_model(self):
        for layer in self.model.layers:
            weights_shape = layer.weights.shape
            print(weights_shape)

model = MLP(1, [5], 1)
trainer = Trainer(model)