from spatialmath.base import bresenham
import matplotlib.pyplot as plt
p = bresenham((2, 4), (10, 10))
plt.plot((2, 10), (4, 10))
plt.plot(p[0], p[1], 'ok')
plt.plot(p[0], p[1], 'k', drawstyle='steps-post')
ax = plt.gca()
ax.grid()