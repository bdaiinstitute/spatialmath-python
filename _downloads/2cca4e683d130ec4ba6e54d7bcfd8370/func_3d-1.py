import matplotlib.pyplot as plt
from spatialmath.base import trplot, transl, rpy2tr
fig = plt.figure(figsize=(10,10))
text_opts = dict(bbox=dict(boxstyle="round",
    fc="w",
    alpha=0.9),
    zorder=20,
    family='monospace',
    fontsize=8,
    verticalalignment='top')
T = transl(2, 1, 1)@ rpy2tr(0, 0, 0)

ax = fig.add_subplot(331, projection='3d')
trplot(T, ax=ax, dims=[0,4])
ax.text(0.5, 0.5, 4.5, "trplot(T)", **text_opts)
ax = fig.add_subplot(332, projection='3d')
trplot(T, ax=ax, dims=[0,4], originsize=0)
ax.text(0.5, 0.5, 4.5, "trplot(T, originsize=0)", **text_opts)
ax = fig.add_subplot(333, projection='3d')
trplot(T, ax=ax, dims=[0,4], style='line')
ax.text(0.5, 0.5, 4.5, "trplot(T, style='line')", **text_opts)
ax = fig.add_subplot(334, projection='3d')
trplot(T, ax=ax, dims=[0,4], axislabel=False)
ax.text(0.5, 0.5, 4.5, "trplot(T, axislabel=False)", **text_opts)
ax = fig.add_subplot(335, projection='3d')
trplot(T, ax=ax, dims=[0,4], width=3)
ax.text(0.5, 0.5, 4.5, "trplot(T, width=3)", **text_opts)
ax = fig.add_subplot(336, projection='3d')
trplot(T, ax=ax, dims=[0,4], frame='B')
ax.text(0.5, 0.5, 4.5, "trplot(T, frame='B')", **text_opts)
ax = fig.add_subplot(337, projection='3d')
trplot(T, ax=ax, dims=[0,4], color='r', textcolor='k')
ax.text(0.5, 0.5, 4.5, "trplot(T, color='r', textcolor='k')", **text_opts)
ax = fig.add_subplot(338, projection='3d')
trplot(T, ax=ax, dims=[0,4], labels=("u", "v", "w"))
ax.text(0.5, 0.5, 4.5, "trplot(T, labels=('u', 'v', 'w'))", **text_opts)
ax = fig.add_subplot(339, projection='3d')
trplot(T, ax=ax, dims=[0,4], style='rviz')
ax.text(0.5, 0.5, 4.5, "trplot(T, style='rviz')", **text_opts)