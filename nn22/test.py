import numpy as np

def save_csv(X, y, suffix):
    df = np.concatenate((X, y), axis=1)
    np.savetxt(f"data/data-{suffix}.csv", df, delimiter=",")
    

X1 = np.random.uniform(-1, 1, size=(100, 3))
y1 = np.array(X1[:, 0] + X1[:, 1] + X1[:, 2]).reshape(-1, 1)
X2 = np.random.uniform(-1, 1, size=(100, 3))
y2 = np.array(X2[:, 0] + X2[:, 1] + X2[:, 2]).reshape(-1, 1)
save_csv(X1, y1, "training")
save_csv(X2, y2, "test")