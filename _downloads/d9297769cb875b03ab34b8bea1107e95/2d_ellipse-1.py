from spatialmath import Ellipse
from spatialmath.base import plotvol2
ax = plotvol2(5)
e = Ellipse(E=np.array([[1, 1], [1, 2]]))
e.plot()
ax.grid()