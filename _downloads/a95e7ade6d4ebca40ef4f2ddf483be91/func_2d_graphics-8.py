from spatialmath.base import plotvol2, plot_point
import numpy as np
p = np.random.uniform(size=(2,10), low=-5, high=5)
ax = plotvol2(5)
plot_point(p, 'r*', '{0}')
ax.grid()