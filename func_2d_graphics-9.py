from spatialmath.base import plotvol2, plot_point
import numpy as np
p = np.random.uniform(size=(2,10), low=-5, high=5)
value = np.random.uniform(size=(10,))
ax = plotvol2(5)
plot_point(p, 'r*', ('{1:.2f}', value))
ax.grid()