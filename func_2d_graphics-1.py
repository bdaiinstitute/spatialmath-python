from spatialmath.base import plotvol2, plot_arrow
ax = plotvol2(5)
plot_arrow((-2, 2), (3, 4), color='r', width=0.1)  # red arrow
ax.grid()