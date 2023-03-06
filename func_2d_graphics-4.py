from spatialmath.base import plotvol2, plot_ellipse
ax = plotvol2(5)
plot_ellipse(np.array([[1, 1], [1, 2]]), [0,0], 'r')  # red ellipse
plot_ellipse(np.array([[1, 1], [1, 2]]), [1, 2], 'b--')  # blue dashed ellipse
plot_ellipse(np.array([[1, 1], [1, 2]]), [-2, -1], filled=True, facecolor='y')  # yellow filled ellipse
ax.grid()