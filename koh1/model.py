import pandas as pd

class SelfOrganizingMap:
    def __init__(self, dataset_name=None, data_X=None, data_y=None):
        self.dataset_name = dataset_name
        self.data_X = data_X
        self.data_y = data_y
        self.read_data()

    def read_data(self):
        if self.dataset_name is None:
            return 
        if self.data_X is not None and self.data_y is not None:
            return
        data_path = f"./data/{self.dataset_name}.csv"
        data = pd.read_csv(data_path)
        if self.data_X is None:
            self.data_X = data.drop("c", axis=1)
        if self.data_y is None:
            self.data_y = data["c"]

    def plot_data(self):
        import matplotlib.pyplot as plt
        plt.scatter(self.data_X["x"], self.data_X["y"], c=self.data_y)
        plt.show()


if __name__ == '__main__':
    som = SelfOrganizingMap(dataset_name="cube")
    som.plot_data()