from spatialmath.base import gauss2d, plotvol3
import matplotlib.pyplot as plt
import numpy as np
a = np.linspace(-5, 5, 100)
x, y = np.meshgrid(a, a)
P = np.diag([1, 2])**2;
g = gauss2d([0, 0], P, x, y)
ax = plotvol3()
ax.plot_surface(x, y, g)