import spatialmath.base as tr
import matplotlib.pyplot as plt

tr.trplot( tr.transl(1,2,3), frame='A', rviz=True, width=1)
tr.trplot( tr.transl(3,1, 2), color='red', width=3, frame='B')
tr.trplot( tr.transl(4, 3, 1)@ tr.trotx(60, 'deg'), color='green', frame='c')

plt.show()
