from spatialmath import Polygon2
from spatialmath.base import plotvol2
p = Polygon2([(1, 2), (3, 2), (2, 4)])
plotvol2(5)
p.plot(fill=False)