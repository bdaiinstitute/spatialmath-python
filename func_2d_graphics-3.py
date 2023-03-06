from spatialmath.base import plotvol2, plot_circle
ax = plotvol2(5)
plot_circle(1, (0,0), 'r')  # red circle
plot_circle(2, (1, 2), 'b--')  # blue dashed circle
plot_circle(0.5, (3,4), filled=True, facecolor='y')  # yellow filled circle
ax.grid()