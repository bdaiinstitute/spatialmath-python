import matplotlib.pyplot as plt
from spatialmath.base import trplot2, transl2, trot2
import math
fig, ax = plt.subplots(3,3, figsize=(10,10))
text_opts = dict(bbox=dict(boxstyle="round", 
    fc="w", 
    alpha=0.9), 
    zorder=20,
    family='monospace',
    fontsize=8,
    verticalalignment='top')
T = transl2(2, 1)@trot2(math.pi/3)
trplot2(T, ax=ax[0][0], dims=[0,4,0,4])
ax[0][0].text(0.2, 3.8, "trplot2(T)", **text_opts)
trplot2(T, ax=ax[0][1], dims=[0,4,0,4], originsize=0)
ax[0][1].text(0.2, 3.8, "trplot2(T, originsize=0)", **text_opts)
trplot2(T, ax=ax[0][2], dims=[0,4,0,4], arrow=False)
ax[0][2].text(0.2, 3.8, "trplot2(T, arrow=False)", **text_opts)
trplot2(T, ax=ax[1][0], dims=[0,4,0,4], axislabel=False)
ax[1][0].text(0.2, 3.8, "trplot2(T, axislabel=False)", **text_opts)
trplot2(T, ax=ax[1][1], dims=[0,4,0,4], width=3)
ax[1][1].text(0.2, 3.8, "trplot2(T, width=3)", **text_opts)
trplot2(T, ax=ax[1][2], dims=[0,4,0,4], frame='B')
ax[1][2].text(0.2, 3.8, "trplot2(T, frame='B')", **text_opts)
trplot2(T, ax=ax[2][0], dims=[0,4,0,4], color='r', textcolor='k')
ax[2][0].text(0.2, 3.8, "trplot2(T, color='r',textcolor='k')", **text_opts)
trplot2(T, ax=ax[2][1], dims=[0,4,0,4], labels=("u", "v"))
ax[2][1].text(0.2, 3.8, "trplot2(T, labels=('u', 'v'))", **text_opts)
trplot2(T, ax=ax[2][2], dims=[0,4,0,4], rviz=True)
ax[2][2].text(0.2, 3.8, "trplot2(T, rviz=True)", **text_opts)