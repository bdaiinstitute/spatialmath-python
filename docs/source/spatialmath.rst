*******
Classes
*******

The top-level classes 

3D-space
========

Pose & orientation: SE(3), SO(3)
--------------------------------

Pose SE(3)
^^^^^^^^^^

.. autoclass:: spatialmath.pose3d.SE3
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :special-members: __mul__,  __truediv__, __add__, __sub__, __eq__, __ne__, __pow__,   __init__
   :exclude-members: count, copy, index, sort, remove

Orientation SO(3)
^^^^^^^^^^^^^^^^^

.. autoclass:: spatialmath.pose3d.SO3
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :special-members: __mul__,  __truediv__, __add__, __sub__, __eq__, __ne__, __pow__,   __init__
   :exclude-members: count, copy, index, sort, remove

Quaternions
-----------

Unit quaternions
^^^^^^^^^^^^^^^^

.. autoclass:: spatialmath.quaternion.UnitQuaternion
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :special-members: __mul__,  __truediv__, __add__, __sub__, __eq__, __ne__, __pow__, __init__
   :exclude-members: count, copy, index, sort, remove

General quaternions
^^^^^^^^^^^^^^^^^^^

.. autoclass:: spatialmath.quaternion.Quaternion
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :special-members: __mul__,  __truediv__, __add__, __sub__, __eq__, __ne__, __pow__, __init__
   :exclude-members: count, copy, index, sort, remove


Twists: se(3)
-------------

.. automodule:: spatialmath.Twist
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :special-members: __mul__,  __truediv__, __add__, __sub__, __eq__, __ne__, __pow__,   __init__
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

Geometry
--------

.. automodule:: spatialmath.geom3d
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :special-members: __mul__, __rmul__, __eq__, __ne__, __init__, __or__, __xor__

2D-space
========

Pose & orientation: SE(2), SO(2)
--------------------------------

Pose SE(2)
^^^^^^^^^^

.. autclass:: spatialmath.pose2d.SE2
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :special-members: __mul__,  __truediv__, __add__, __sub__, __eq__, __ne__, __pow__,   __init__
   :exclude-members: count, copy, index, sort, remove

Orientation SO(2)
^^^^^^^^^^^^^^^^^

.. autclass:: spatialmath.pose2d.SO2
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :special-members: __mul__,  __truediv__, __add__, __sub__, __eq__, __ne__, __pow__,   __init__
   :exclude-members: count, copy, index, sort, remove

Twists se(2)
------------

.. automodule:: spatialmath.Twist2
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :special-members: __mul__,  __truediv__, __add__, __sub__, __eq__, __ne__, __pow__,   __init__
   :exclude-members: count, copy, index, sort, remove
   
Geometry
--------

.. automodule:: spatialmath.geom2d
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :special-members: __mul__, __rmul__, __eq__, __ne__, __init__, __or__, __xor__

***********************
Function library (base)
***********************

blah blah blah

Transforms in 2D
================

.. automodule:: spatialmath.base.transforms2d
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :special-members:
   
Transforms in 3D
================

.. automodule:: spatialmath.base.transforms3d
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :special-members:


Transforms in ND
================

.. automodule:: spatialmath.base.transformsNd
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :special-members:

Quaternions
===========

.. automodule:: spatialmath.base.quaternions
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :special-members:

Utility
=======

Vectors
-------

.. automodule:: spatialmath.base.vectors
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :special-members:

Graphic animation
-----------------

.. automodule:: spatialmath.base.vectors
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :special-members:

Argument checking
-----------------

.. automodule:: spatialmath.base.argcheck
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :special-members:

