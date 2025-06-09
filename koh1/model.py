from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import random
from matplotlib.widgets import Button, Slider
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
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

        def draw_frame(i):
            nonlocal scatter
            ax.clear()
            m = self.history[i]
            coords = m['map']
            if self.map.data_dim == 2:
                if show_data:
                    ax.scatter(self.map.data_X.iloc[:, 0], self.map.data_X.iloc[:, 1], c=self.map.data_y, s=10)
                scatter = ax.scatter(coords[:, :, 0].flatten(), coords[:, :, 1].flatten(), facecolors='none', edgecolors='red', s=30)
                for i in range(self.map.width-1):
                    for j in range(self.map.height-1):
                        ax.plot([coords[i,j,0], coords[i+1,j,0]], [coords[i,j,1], coords[i+1,j,1]], c='black', lw=0.5)
                        ax.plot([coords[i,j,0], coords[i,j+1,0]], [coords[i,j,1], coords[i,j+1,1]], c='black', lw=0.5)
                ax.set_xlim(self.map.ranges[0][0], self.map.ranges[0][1])
                ax.set_ylim(self.map.ranges[1][0], self.map.ranges[1][1])
            else:
                if show_data:
                    ax.scatter(self.map.data_X.iloc[:, 0], self.map.data_X.iloc[:, 1], self.map.data_X.iloc[:, 2], c=self.map.data_y, s=10)
                scatter = ax.scatter(coords[:, :, 0].flatten(), coords[:, :, 1].flatten(), coords[:, :, 2].flatten(),
                                    facecolors='none', edgecolors='red', s=30)
                for i in range(self.map.width-1):
                    for j in range(self.map.height-1):
                        ax.plot3D([coords[i,j,0], coords[i+1,j,0]], [coords[i,j,1], coords[i+1,j,1]], [coords[i,j,2], coords[i+1,j,2]], c='black', lw=0.5)
                        ax.plot3D([coords[i,j,0], coords[i,j+1,0]], [coords[i,j,1], coords[i,j+1,1]], [coords[i,j,2], coords[i,j+1,2]], c='black', lw=0.5)
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

    def _init_map(self):
        self.ranges = []
        for dim in range(self.data_dim):
            self.ranges.append((self.data_X.iloc[:,dim].min(), self.data_X.iloc[:,dim].max()))
        
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
            for index, data_point in enumerate(self.data_X.sample(frac=1).values):
                bmu = self._find_bmu(data_point)
                self.move_map(bmu, data_point, alpha, sigma, proximity_function)
                self.map_detail_history.add(self.map, epoch, index)
            self.map_history.add(self.map, epoch)

            if visualize:
                self.plot()

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

    def plot_data(self, show=True):
        if self.data_dim == 2:
            plt.scatter(self.data_X.iloc[:, 0], self.data_X.iloc[:, 1], c=self.data_y)
        if self.data_dim > 2:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.data_X.iloc[:, 0], self.data_X.iloc[:, 1], self.data_X.iloc[:, 2], c=self.data_y)
        if show:
            plt.show()
        return plt

    def plot(self, show=True):
        if self.data_dim == 2:
            plt.scatter(self.data_X.iloc[:, 0], self.data_X.iloc[:, 1], c=self.data_y, s=10)
            plt.scatter(self.map[:,:,0].flatten(), self.map[:,:,1].flatten(), facecolors='none', edgecolors='red', s=5)

            for i in range(self.width-1):
                for j in range(self.height-1):
                    plt.plot([self.map[i,j,0], self.map[i+1,j,0]], [self.map[i,j,1], self.map[i+1,j,1]], c='black', lw=0.5)
                    plt.plot([self.map[i,j,0], self.map[i,j+1,0]], [self.map[i,j,1], self.map[i,j+1,1]], c='black', lw=0.5)
            if show:
                plt.show()
            return plt.gca()

        if self.data_dim > 2:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.data_X.iloc[:, 0], self.data_X.iloc[:, 1], self.data_X.iloc[:, 2], c=self.data_y, s=10)
            ax.scatter(self.map[:,:,0].flatten(), self.map[:,:,1].flatten(), self.map[:,:,2].flatten(), facecolors='none', edgecolors='red', s=5)

            for i in range(self.width-1):
                for j in range(self.height-1):
                    ax.plot3D([self.map[i,j,0], self.map[i+1,j,0]], [self.map[i,j,1], self.map[i+1,j,1]], [self.map[i,j,2], self.map[i+1,j,2]], c='black', lw=0.5)
                    ax.plot3D([self.map[i,j,0], self.map[i,j+1,0]], [self.map[i,j,1], self.map[i,j+1,1]], [self.map[i,j,2], self.map[i,j+1,2]], c='black', lw=0.5)
            if show:
                plt.show()
            return ax

    def plot_point_and_bmu(self, point, bmu, show=True):
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


if __name__ == '__main__':
    som = SelfOrganizingMap(dataset_name="cube", width=15, height=15)
    # som.train(epochs=20, visualize=True)
    # som.train(epochs=10, proximity_function="neg_second_gaussian_derivative")
    som.train(epochs=15)
    som.plot(show=True)
    som.show_history(delay=0.3, show_data=True)