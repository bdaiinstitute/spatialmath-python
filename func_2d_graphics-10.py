from spatialmath.base import plotvol2, plot_text
ax = plotvol2(5)
plot_text((0,0), 'foo')
plot_text((1,1), 'bar', color='b')
plot_text((2,2), 'baz', fontsize=14, horizontalalignment='center')
ax.grid()