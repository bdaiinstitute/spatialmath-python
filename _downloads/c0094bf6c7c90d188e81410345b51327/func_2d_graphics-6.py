from spatialmath.base import plotvol2, plot_homline
ax = plotvol2(5)
plot_homline((1, -2, 3))
plot_homline((1, -2, 3), 'k--') # dashed black line
ax.grid()