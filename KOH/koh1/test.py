import matplotlib.pyplot as plt
import numpy as np
import time

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for _ in range(5):
    ax.clear()
    points = np.random.rand(100, 3)
    ax.scatter(points[:,0], points[:,1], points[:,2], facecolors='none', edgecolors='red', s=5)
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.5)

plt.ioff()
plt.show()