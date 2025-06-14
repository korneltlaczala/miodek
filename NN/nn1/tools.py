import matplotlib.pyplot as plt
import pandas as pd


def plot(filename: str):
    df = pd.read_csv(filename)
    print(type(df['x']))
    print(f"Number of rows: {len(df)}")
    plt.scatter(df['x'], df['y'])
    plt.show()

plot("data/steps-large-training.csv")
plot("data/square-simple-training.csv")