import numpy as np

def save_csv(X, suffix):
    np.savetxt(f"data/data-{suffix}.csv", X, delimiter=",")

X1 = np.random.uniform(-1, 1, size=(100, 3))
X2 = np.random.uniform(-1, 1, size=(100, 3))
save_csv(X1, "training")
save_csv(X2, "test")