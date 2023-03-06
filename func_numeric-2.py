from spatialmath.base import gauss1d
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0, 10, 100)
g = gauss1d(5, 2, x)
plt.plot(x, g)
plt.grid()