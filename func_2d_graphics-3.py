from spatialmath.base import plotvol2, plot_box
ax = plotvol2(5)
plot_box("b--", centre=(2, 3), wh=1)  # w=h=1
plot_box(lt=(0, 0), rb=(3, -2), filled=True, hatch="/", edgecolor="k", color="r")
ax.grid()