from spatialmath.base import plot_ellipsoid, plotvol3
import numpy as np

plotvol3(4)
plot_ellipsoid(np.diag([1, 2, 3]), [1, 1, 0], color="r", resolution=5); # draw red ellipsoid