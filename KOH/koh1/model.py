from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import random
from matplotlib.widgets import Button, Slider
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D

class MapHistory():
    def __init__(self, map):
        self.history = []
        self.map = map

    def add(self, map, epoch=None, iteration=None):
        self.history.append({
            "map": map.copy(),
            "epoch": epoch,
            "iteration": iteration,
        })

    def animate(self, delay=0.5, show_data=False):
        from mpl_toolkits.mplot3d import Axes3D  # needed for 3D projection
        import numpy as np

        # Prepare figure
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d' if self.map.data_dim > 2 else None)

        plt.subplots_adjust(bottom=0.25)  # Make space for slider and button

        # Initial plot
        scatter = None
        title = ax.set_title("")

        def draw_frame(frame_nr):
            nonlocal scatter
            ax.clear()
            m = self.history[frame_nr]
            coords = m['map']
            if self.map.data_dim == 2:
                if show_data:
                    ax.scatter(self.map.data_X.iloc[:, 0], self.map.data_X.iloc[:, 1], c=self.map.data_y, s=10)
                scatter = ax.scatter(coords[:, :, 0].flatten(), coords[:, :, 1].flatten(), facecolors='none', edgecolors='red', s=30)
                for i in range(self.map.width):
                    for j in range(self.map.height):
                        ax.plot([coords[i,j,0], coords[i+1,j,0]], [coords[i,j,1], coords[i+1,j,1]], c='black', lw=0.5) if i < self.map.width-1 else None
                        ax.plot([coords[i,j,0], coords[i,j+1,0]], [coords[i,j,1], coords[i,j+1,1]], c='black', lw=0.5) if j < self.map.height-1 else None
                ax.set_xlim(self.map.ranges[0][0], self.map.ranges[0][1])
                ax.set_ylim(self.map.ranges[1][0], self.map.ranges[1][1])
            else:
                if show_data:
                    ax.scatter(self.map.data_X.iloc[:, 0], self.map.data_X.iloc[:, 1], self.map.data_X.iloc[:, 2], c=self.map.data_y, s=10)
                scatter = ax.scatter(coords[:, :, 0].flatten(), coords[:, :, 1].flatten(), coords[:, :, 2].flatten(),
                                    facecolors='none', edgecolors='red', s=30)
                for i in range(self.map.width):
                    for j in range(self.map.height):
                        ax.plot3D([coords[i,j,0], coords[i+1,j,0]], [coords[i,j,1], coords[i+1,j,1]], [coords[i,j,2], coords[i+1,j,2]], c='black', lw=0.5) if i < self.map.width - 1 else None
                        ax.plot3D([coords[i,j,0], coords[i,j+1,0]], [coords[i,j,1], coords[i,j+1,1]], [coords[i,j,2], coords[i,j+1,2]], c='black', lw=0.5) if j < self.map.height - 1 else None
                ax.set_xlim(self.map.ranges[0][0], self.map.ranges[0][1])
                ax.set_ylim(self.map.ranges[1][0], self.map.ranges[1][1])
                ax.set_zlim(self.map.ranges[2][0], self.map.ranges[2][1])

            ax.set_title(f"Epoch: {m['epoch']+1}, Iteration: {m['iteration']}")

        # Draw the first frame
        draw_frame(0)

        # Slider
        ax_slider = plt.axes([0.15, 0.1, 0.7, 0.03])
        slider = Slider(ax_slider, 'Epoch', 0, len(self.history) - 1, valinit=0, valstep=1)

        # Play/Pause button
        ax_button = plt.axes([0.85, 0.02, 0.1, 0.04])
        button = Button(ax_button, 'Pause')

        is_paused = False

        def on_slider_change(val):
            draw_frame(int(val))
            fig.canvas.draw_idle()

        slider.on_changed(on_slider_change)

        def on_button_clicked(event):
            nonlocal is_paused
            is_paused = not is_paused
            button.label.set_text('Play' if is_paused else 'Pause')

        button.on_clicked(on_button_clicked)

        # Automatic animation
        def update(frame):
            if not is_paused:
                new_val = (int(slider.val) + 1) % len(self.history)
                slider.set_val(new_val)

        anim = FuncAnimation(fig, update, interval=delay * 1000)

        plt.show()

class SelfOrganizingMap:
    def __init__(self, dataset_name=None, data_X=None, data_y=None, width=10, height=10):
        self.dataset_name = dataset_name
        self.data_X = data_X
        self.data_y = data_y
        self.width = width
        self.height = height
        self._read_data()
        self._init_map()
        self.map_detail_history = MapHistory(self)
        self.map_history = MapHistory(self)

    def _read_data(self):
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

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data_X, self.data_y, test_size=0.2, random_state=42, stratify=self.data_y
        )

    def get_xy_for_data_type(self, data_type="test"):
        if data_type == "test":
            return self.X_test, self.y_test
        elif data_type == "train":
            return self.X_train, self.y_train
        elif data_type == "all":
            return self.data_X, self.data_y
        elif data_type == "empty":
            return pd.DataFrame(columns=self.data_X.columns), pd.Series(dtype=self.data_y.dtype)
        else:
            raise ValueError("data_type must be 'test', 'train', or 'all'")

    def _init_map(self):
        self.ranges = []
        for dim in range(self.data_dim):
            self.ranges.append((self.X_train.iloc[:,dim].min(), self.X_train.iloc[:,dim].max()))
        
        x = np.linspace(self.ranges[0][0], self.ranges[0][1], self.width)
        y = np.linspace(self.ranges[1][0], self.ranges[1][1], self.height)
        self.map = np.zeros((self.width, self.height, self.data_dim))
        for i in range(self.width):
            for j in range(self.height):
                vector = np.array([x[i], y[j]])
                for dim in range(2, self.data_dim):
                    vector = np.append(vector, random.uniform(self.ranges[dim][0], self.ranges[dim][1]))
                self.map[i][j] = vector
                      

    def train(self, epochs, lambda_decay=10, sigma=1, proximity_function="gaussian", verbose=True, visualize=False):
        iterator = (
            tqdm(range(epochs), ncols=100, colour='green') if verbose else
            range(epochs)
        )
        for epoch in iterator:
            alpha = np.exp(-epoch/lambda_decay)
            for index, data_point in enumerate(self.X_train.sample(frac=1).values):
                bmu = self._find_bmu(data_point)
                self.move_map(bmu, data_point, alpha, sigma, proximity_function)
                self.map_detail_history.add(self.map, epoch, index)
            self.map_history.add(self.map, epoch)

            if visualize:
                self.plot()
        self.classify()

    def move_map(self, bmu, data_point, alpha, sigma, proximity_function):
        for i in range(self.width):
            for j in range(self.height):
                distance = np.linalg.norm(np.array([i, j]) - np.array(bmu))
                self.map[i][j] += alpha * 1/10 * (data_point - self.map[i][j]) * self.proximity_coeff(distance, sigma, proximity_function)

    def proximity_coeff(self, distance, sigma, function):
        if function == "gaussian":
            return np.exp(-distance**2 / (2 * sigma**2))
        if function == "neg_second_gaussian_derivative":
            return distance * np.exp(-distance**2 / (2 * sigma**2))

    def _find_bmu(self, data_point):
        distances = np.linalg.norm(self.map - data_point, axis=2)
        return np.unravel_index(distances.argmin(), distances.shape)

    def demonstrate_bmu(self):
        data_point = self.data_X.sample(frac=1).values[0]
        bmu = self._find_bmu(data_point)
        self.plot_point_and_bmu(data_point, bmu)

    def plot_data(self, show=True, data_type="test"):
        X, y = self.get_xy_for_data_type(data_type)
        if self.data_dim == 2:
            plt.figure(figsize=(10, 8))
            plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y)
        if self.data_dim > 2:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X.iloc[:, 0], X.iloc[:, 1], X.iloc[:, 2], c=y)
        if show:
            plt.show()
            return
        return plt

    def plot_data_split(self, show=True):
        fig, axs = plt.subplots(1, 2, figsize=(16, 6))
        plt.tight_layout()
        # Plot train data
        X_train, y_train = self.get_xy_for_data_type("train")
        if self.data_dim == 2:
            axs[0].scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=y_train, s=10)
            axs[0].set_title("Train Data")
        else:
            ax = fig.add_subplot(121, projection='3d')
            ax.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], X_train.iloc[:, 2], c=y_train, s=10)
            ax.set_title("Train Data")
            axs[0].remove()

        # Plot test data
        X_test, y_test = self.get_xy_for_data_type("test")
        if self.data_dim == 2:
            axs[1].scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_test, s=10)
            axs[1].set_title("Test Data")
        else:
            ax = fig.add_subplot(122, projection='3d')
            ax.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], X_test.iloc[:, 2], c=y_test, s=10)
            ax.set_title("Test Data")
            axs[1].remove()


        if show:
            plt.show()
        else:
            return plt

    def plot(self, show=True, data_type="test", show_net=True, show_mislabelled=False):
        X, y = self.get_xy_for_data_type(data_type)
        if show_mislabelled:
            if not hasattr(self, 'classification_map'):
                raise ValueError("Please run classify() method before plotting mislabelled points")
            y_pred = self.predict(X)
            mislabelled = []
            for data_point, true_class, predicted_class in zip(X.values, y.values, y_pred):
                if predicted_class != true_class:
                    mislabelled.append(data_point)
            mislabelled = np.array(mislabelled)
            y = y_pred

        if self.data_dim == 2:
            plt.figure(figsize=(10, 8))
            plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, s=20)
            if show_mislabelled:
                plt.scatter(mislabelled[:,0].flatten(), mislabelled[:,1].flatten(), marker='x', color='red', s=80, label='Mislabelled')
            if show_net:
                plt.scatter(self.map[:,:,0].flatten(), self.map[:,:,1].flatten(), facecolors='none', edgecolors='red', s=5)
                for i in range(self.width):
                    for j in range(self.height):
                        plt.plot([self.map[i,j,0], self.map[i+1,j,0]], [self.map[i,j,1], self.map[i+1,j,1]], c='dimgray', lw=0.5) if i < self.width-1 else None
                        plt.plot([self.map[i,j,0], self.map[i,j+1,0]], [self.map[i,j,1], self.map[i,j+1,1]], c='dimgray', lw=0.5) if j < self.height-1 else None
            if show == False:
                return plt.gca()

        if self.data_dim > 2:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X.iloc[:, 0], X.iloc[:, 1], X.iloc[:, 2], c=y, s=10)
            if show_mislabelled:
                ax.scatter(mislabelled[:,0].flatten(), mislabelled[:,1].flatten(), mislabelled[:,2].flatten(), marker='x', color='red', s=80, label='Mislabelled')
            if show_net:
                ax.scatter(self.map[:,:,0].flatten(), self.map[:,:,1].flatten(), self.map[:,:,2].flatten(), facecolors='none', edgecolors='red', s=5)
                for i in range(self.width):
                    for j in range(self.height):
                        if i < self.width - 1:
                            ax.plot3D(
                                [self.map[i, j, 0], self.map[i + 1, j, 0]],
                                [self.map[i, j, 1], self.map[i + 1, j, 1]],
                                [self.map[i, j, 2], self.map[i + 1, j, 2]],
                                c='dimgray', lw=0.5
                            )
                        if j < self.height - 1:
                            ax.plot3D(
                                [self.map[i, j, 0], self.map[i, j + 1, 0]],
                                [self.map[i, j, 1], self.map[i, j + 1, 1]],
                                [self.map[i, j, 2], self.map[i, j + 1, 2]],
                                c='dimgray', lw=0.5
                            )
            if show == False:
                return ax

    def plot_net_classification(self, show=True):
        if not hasattr(self, 'classification_map'):
            raise ValueError("Please run classify() method before plotting net classification")
        
        if self.data_dim == 2:
            plt.figure(figsize=(10, 8))
            plt.scatter(self.map[:, :, 0].flatten(), self.map[:, :, 1].flatten(), c=self.classification_map.flatten(), s=40)
            for i in range(self.width):
                for j in range(self.height):
                    if i < self.width - 1:
                        plt.plot([self.map[i, j, 0], self.map[i + 1, j, 0]],
                                 [self.map[i, j, 1], self.map[i + 1, j, 1]],
                                 c='dimgray', lw=0.5)
                    if j < self.height - 1:
                        plt.plot([self.map[i, j, 0], self.map[i, j + 1, 0]],
                                 [self.map[i, j, 1], self.map[i, j + 1, 1]],
                                 c='dimgray', lw=0.5)
            plt.title("SOM Net Classification")
            if show:
                plt.show()
            else:
                return plt.gca()
        else:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.map[:, :, 0].flatten(), self.map[:, :, 1].flatten(), self.map[:, :, 2].flatten(),
                        s=40, c=self.classification_map.flatten())
            for i in range(self.width):
                for j in range(self.height):
                    if i < self.width - 1:
                        ax.plot3D([self.map[i, j, 0], self.map[i + 1, j, 0]],
                                  [self.map[i, j, 1], self.map[i + 1, j, 1]],
                                  [self.map[i, j, 2], self.map[i + 1, j, 2]],
                                  c='dimgray', lw=0.5)
                    if j < self.height - 1:
                        ax.plot3D([self.map[i, j, 0], self.map[i, j + 1, 0]],
                                  [self.map[i, j, 1], self.map[i, j + 1, 1]],
                                  [self.map[i, j, 2], self.map[i, j + 1, 2]],
                                  c='dimgray', lw=0.5)
            ax.set_title("SOM Net Classification")
            if show:
                plt.show()
            else:
                return ax

    def plot_point_and_bmu(self, point=None, bmu=None, show=True):
        if point is None:
            point = self.X_test.sample(frac=1).values[0]
        if bmu is None:
            bmu = self._find_bmu(point)

        if self.data_dim == 2:
            plt = self.plot(show=False)
            plt.scatter(point[0], point[1], facecolors="none", edgecolors="black", s=30)
            plt.scatter(self.map[bmu[0], bmu[1], 0], self.map[bmu[0], bmu[1], 1], facecolors="none", edgecolors="black", s=30)

        if self.data_dim > 2:
            ax = self.plot(show=False)
            ax.scatter(point[0], point[1], point[2], facecolors="none", edgecolors="black", s=30)
            ax.scatter(self.map[bmu[0], bmu[1], 0], self.map[bmu[0], bmu[1], 1], self.map[bmu[0], bmu[1], 2], facecolors="none", edgecolors="black", s=30)
                
        if show:
            import matplotlib.pyplot as plt
            plt.show()

    def show_history(self, delay=0.5, show_data=False):
        self.map_history.animate(delay=delay, show_data=show_data)

    def generate_projection(self, data_type="test"):
        data_X, data_y = self.get_xy_for_data_type(data_type)
        x = []
        y = []
        for data_point in data_X.values:
            bmu = self._find_bmu(data_point)
            x.append(bmu[0] + (random.random() - 0.5) * 0.5)
            y.append(bmu[1] + (random.random() - 0.5) * 0.5)
        return x, y, data_y

    def plot_projection(self, data_type="test", colored_neurons=False):
        if colored_neurons and not hasattr(self, 'classification_map'):
            raise ValueError("Please run classify() method before plotting colored neurons")
        colors = self.classification_map.flatten() if colored_neurons else "black"
        plt.scatter([[i for j in range(self.height)] for i in range(self.width)],
                    [[j for j in range(self.height)] for i in range(self.width)],
                    c=colors, s=50, edgecolors='black', alpha=0.8, label='Neurons')

        x, y, data_y = self.generate_projection(data_type=data_type)
        plt.scatter(x, y, c=data_y, edgecolors="#222", s=20)
        plt.show()

    def predict(self, data_X=None):
        if not hasattr(self, 'classification_map'):
            raise ValueError("Please run classify() method before predict()")
        if data_X is None:
            data_X = self.X_test

        predictions = []
        for data_point in data_X.values:
            bmu = self._find_bmu(data_point)
            predicted_class = self.classification_map[bmu[0], bmu[1]]
            predictions.append(predicted_class)
        return predictions

    def classify(self):
        votes = np.zeros((self.width, self.height, len(self.data_y.unique())), dtype=int)
        for data_point, true_class in zip(self.X_train.values, self.y_train.values):
            bmu = self._find_bmu(data_point)
            votes[bmu[0], bmu[1], true_class] += 1
        self.classification_map = np.argmax(votes, axis=2)

    def performance_summary(self, data_type="test"):
        X, y = self.get_xy_for_data_type(data_type)
        if not hasattr(self, 'classification_map'):
            raise ValueError("Please run classify() method before performance_summary()")
        
        y_pred = self.predict(X)
        correct_predictions = sum(p == t for p, t in zip(y_pred, y.values))
        total_predictions = len(y_pred)

        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        print(f"=== Performance Summary ===")
        print(f"Total Predictions: {total_predictions}")
        print(f"Correct Predictions: {correct_predictions}")
        print(f"Wrong Predictions: {total_predictions - correct_predictions}")
        print(f"Accuracy: {accuracy:.2f}")


if __name__ == '__main__':
    som = SelfOrganizingMap(dataset_name="hexagon", width=10, height=10)
    som.train(epochs=100, proximity_function="neg_second_gaussian_derivative", sigma=0.4, lambda_decay=1000)
    # som.plot_projection()
    som.plot(show=True)
    som.show_history(delay=0.2, show_data=True)