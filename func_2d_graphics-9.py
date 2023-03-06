from spatialmath.base import plotvol2, plot_polygon
ax = plotvol2(5)
vertices = np.array([[-1, 2, -1], [1, 0, -1]])
plot_polygon(vertices, filled=True, facecolor='g')  # green filled triangle
ax.grid()