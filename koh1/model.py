import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

class SelfOrganizingMap:
    def __init__(self, dataset_name=None, data_X=None, data_y=None):
        self.dataset_name = dataset_name
        self.data_X = data_X
        self.data_y = data_y
        self.read_data()

    def read_data(self):
        if self.dataset_name is None:
            if self.data_X is None or self.data_y is None:
                raise ValueError("Please provide either dataset_name or data_X and data_y")
            return
        if self.data_X is None or self.data_y is None:
            data_path = f"./data/{self.dataset_name}.csv"
            data = pd.read_csv(data_path)
        if self.data_X is None:
            self.data_X = data.drop("c", axis=1)
        if self.data_y is None:
            self.data_y = data["c"]
        self.data_dim = self.data_X.shape[1]

    def plot_data(self):
        import matplotlib.pyplot as plt
        if self.data_dim == 2:
            plt.scatter(self.data_X.iloc[:, 0], self.data_X.iloc[:, 1], c=self.data_y)
        if self.data_dim > 2:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.data_X.iloc[:, 0], self.data_X.iloc[:, 1], self.data_X.iloc[:, 2], c=self.data_y)
        plt.show()


if __name__ == '__main__':
    som = SelfOrganizingMap(dataset_name="cube")
    som.plot_data()