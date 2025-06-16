from models import *

model = MLP(
    architecture=MLPArchitecture(1, [10, 10], 1),
    dataset_name="multimodal-large",
)

model.train(
    epochs=1000,
    learning_rate=0.1,
    batch=True,
    batch_size=32,
    optimizer="rmsprop",
    rms_beta=0.9,
    # regularization="l1",
)
model.plot_fit_and_history()
