3D-space
========

Pose in 3D
----------

SE(3) matrix
^^^^^^^^^^^^

.. autoclass:: spatialmath.pose3d.SE3
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :special-members: __mul__,  __truediv__, __add__, __sub__, __eq__, __ne__, __pow__,   __init__
   :exclude-members: count, copy, index, sort, remove, arghandler, binop, unop

se(3) twist 
^^^^^^^^^^^

.. automodule:: spatialmath.Twist3
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :special-members: __mul__,  __truediv__, __add__, __sub__, __eq__, __ne__, __pow__,   __init__
   :exclude-members: count, copy, index, sort, remove

Orientation in 3D
-----------------

SO(3) matrix
^^^^^^^^^^^^

.. autoclass:: spatialmath.pose3d.SO3
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :special-members: __mul__,  __truediv__, __add__, __sub__, __eq__, __ne__, __pow__,   __init__
   :exclude-members: count, copy, index, sort, remove

Unit quaternion
^^^^^^^^^^^^^^^

.. autoclass:: spatialmath.quaternion.UnitQuaternion
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :special-members: __mul__,  __truediv__, __add__, __sub__, __eq__, __ne__, __pow__, __init__
   :exclude-members: count, copy, index, sort, remove


6D spatial vectors
------------------

.. automodule:: spatialmath.spatialvector
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :special-members: __mul__,  __truediv__, __add__, __sub__, __eq__, __ne__, __pow__,   __init__
   :exclude-members: count, copy, index, sort, remove


General quaternions
-------------------

.. autoclass:: spatialmath.quaternion.Quaternion
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :special-members: __mul__,  __truediv__, __add__, __sub__, __eq__, __ne__, __pow__, __init__
   :exclude-members: count, copy, index, sort, remove

Geometry
--------

.. automodule:: spatialmath.geom3d
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :special-members: __mul__, __rmul__, __eq__, __ne__, __init__, __or__, __xor__