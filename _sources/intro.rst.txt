
************
Introduction
************


Spatial maths capability underpins all of robotics and robotic vision. 
It provides the means to describe the relative position and orientation of objects in 2D or 3D space.  
This package provides Python classes and functions to represent, print, plot, manipulate and covert between such representations.
This includes relevant mathematical objects such as rotation matrices :math:`R \in SO(2), SO(3)`, 
homogeneous transformation matrices :math:`T \in SE(2), SE(3)`, quaternions :math:`q \in \mathbb{H}`,
and twists :math:`t in se(2), se(3)`.

For example, we can create a rigid-body transformation that is a rotation about the x-axis of 30 degrees::

  >>> from spatialmath import *
  >>> base.rotx(30, 'deg')
  array([[ 1.       ,  0.       ,  0.       ],
        [ 0.       ,  0.8660254, -0.5      ],
        [ 0.       ,  0.5      ,  0.8660254]])

which results in a NumPy 4x4 array that belongs to the group SE(3).  We could also create a class instance::

  >>> from spatialmath import *
  >>> T = SE3.Rx(30, 'deg')
  >>> type(T)
  <class 'spatialmath.pose3d.SE3'>
  >>> print(T)
    1           0           0           0            
    0           0.866025   -0.5         0            
    0           0.5         0.866025    0            
    0           0           0           1                  

which is *internally* represented as a 4x4 NumPy array.

While functions and classes can provide similar functionality the class provide the benefits of:

- type safety, it is not possible to mix a 3D rotation matrix with a 2D rigid-body motion, even though both are represented
  by a 3x3 matrix
- operator overloading allows for convenient and readable expression of algorithms
- representing not a just a single value, but a sequence, which are handled by the operators with implicit broadcasting of values


Relationship to MATLAB tools
----------------------------
This package replicates, as much as possible, the functionality of the `Spatial Math Toolbox  <https://github.com/petercorke/spatial-math>`__ for MATLAB |reg| 
which underpins the `Robotics Toolbox <https://github.com/petercorke/robotics-toolbox-matlab>`__ for MATLAB. It comprises:

* the *classic* functions (which date back to the origin of the Robotics Toolbox for MATLAB) such as ``rotx``, ``trotz``, ``eul2tr`` etc. as the ``base`` package which you can access by::

    from spatialmath.base import *

  and works with NumPy arrays.  This package also includes a set of functions to deal with quaternions and unit-quaternions represented as 4-element
  Numpy arrays.
* the classes (which appeared in Robotics Toolbox for MATLAB release 10 in 2017) such as ``SE3``, ``UnitQuaternion`` etc.  The only significant difference
  is that the MATLAB ``Twist`` class is now called ``Twist3``.

The design considerations included:

  - being as similar as possible to the MATLAB Toolbox function names and semantics
  - but balancing the tension of being as Pythonic as possible
  - use Python keyword arguments to replace the MATLAB Toolbox string options supported using ``tb_optparse()``
  - use NumPy arrays internally to represent for rotation and homogeneous transformation matrices, quaternions and vectors
  - all functions that accept a vector can accept a list, tuple, or ``np.ndarray``
  - A class instance can hold a sequence of elements, they are polymorphic with lists, which can be used to represent trajectories or time sequences
  - Classes are _fairly_ polymorphic, they share many common constructor options and methods


Spatial math classes
====================

The package provides classes to represent pose and orientation in 3D and 2D
space:

============  ==================  =============
Represents    in 3D               in 2D
============  ==================  =============
pose          ``SE3``             ``SE2``
              ``Twist3``          ``Twist2``
orientation   ``SO3``             ``SO2``
              ``UnitQuaternion``  
============  ==================  =============

These classes to abstract and implementing appropriate operations for the following groups:

======================  ============================    =======================
Group                   Name                            Class
======================  ============================    =======================
:math:`\mbox{SE(3)}`    rigid-body translation in 3D    ``SE3``
:math:`\mbox{SO(3)}`    orientation in 3D               ``SO3``
:math:`S^3`             unit quaternion                 ``UnitQuaternion``
:math:`\mbox{SE(2)}`    rigid-body translation in 2D    ``SE2``
:math:`\mbox{SO(2)}`    orientation in 2D               ``SO2``
:math:`\mbox{se(3)}`    twist in 3D                     ``Twist3``
:math:`\mbox{se(2)}`    twist in 2D                     ``Twist2``
:math:`\mathbb{H}`      quaternion                      ``Quaternion``
:math:`M^6`             spatial velocity                ``SpatialVelocity``
:math:`M^6`             spatial acceleration            ``SpatialAcceleration``
:math:`F^6`             spatial force                   ``SpatialForce``
:math:`F^6`             spatial momentum                ``SpatialMomentum``
|                       spatial inertia                 ``SpatialInertia``
======================  ============================    =======================


In addition to the merits of classes outlined above, classes ensure that the numerical value is always valid because the 
constraints (eg. orthogonality, unit norm) are enforced when the object is constructed.  For example::

  >>> SE3(np.zeros((4,4)))
  Traceback (most recent call last):
    .
    .
  AssertionError: array must have valid value for the class

Type safety and type validity are particularly important when we deal with a sequence of values.  
In robotics we frequently deal with trajectories of poses or rotation to describe objects moving in the
world.
However a list of these items::

  >>> X = [SE3.Rx(0), SE3.Rx(0.2), SE3.Rx(0.4), SE3.Rx(0.6)]

has the type `list` and the elements are not guaranteed to be homogeneous, ie. a list could contain a mixture of classes.
This requires careful coding, or additional user code to check the validity of all elements in the list.
We could create a NumPy array of these objects, the upside being it could be more than one-dimensional, but the again NumPy does not
enforce homogeneity of object types in an array.

The Spatial Math package give these classes list *super powers* so that, for example, a single `SE3` object can contain a sequence of SE(3) values::

  >>> X = SE3.Rx([0, 0.2, 0.4, 0.6])
  >>> len(x)
  4
  >>> print(X[1])
    1           0           0           0            
    0           0.980067   -0.198669    0            
    0           0.198669    0.980067    0            
    0           0           0           1            

The classes inherit from ``collections.UserList`` and have all the functionality of Python lists, and this is discussed further in
section :ref:`list-powers`
The pose objects are a list subclass so we can index it or slice it as we
would a list, but the result each time belongs to the class it was sliced from.  


Operators for pose objects
--------------------------

Group operations
^^^^^^^^^^^^^^^^

The classes represent mathematical groups, and the group arithmetic rules are enforced.
The operator ``*`` denotes composition and the result will be of the same type as the operand::

  >>> T = SE3.Rx(0.3)
  >>> type(T)
  <class 'spatialmath.pose3d.SE3'>
  >>> X = T * T
  >>> type(X)
  <class 'spatialmath.pose3d.SE3'>

The implementation depends on the class:

* for SO(n) and SE(n) composition is implemented by matrix multiplication of the underlying matrix values,
* for unit-quaternions composition is implemented by the Hamilton product of the underlying vector value,
* for twists it is the logarithm of the product of exponentiating the two twists

``**`` denotes repeated composition so the exponent must be an integer.  If the negative exponent the repeated multiplication
is performed then the inverse is taken.

The group inverse is given by the ``inv()`` method::

  >>> T * T.inv()
  SE3(array([[1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]]))

and ``/`` denotes multiplication by the inverse::

  >>> T / T
  SE3(array([[1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]]))



Constructors
^^^^^^^^^^^^

For every group the identity value can be constructed by instantiating the class with no arguments::

    >>> UnitQuaternion()
    1.000000 << 0.000000, 0.000000, 0.000000 >>

    >>> SE3()
    SE3(array([[1., 0., 0., 0.],
              [0., 1., 0., 0.],
              [0., 0., 1., 0.],
              [0., 0., 0., 1.]]))

Other constructors are implemented as class methods and are common to ``SE3``, ``SO3``, ``SE2``, ``SO2`` and ``UnitQuaternion``
and begin with an uppercase letter:

-----------   --------------------------------------------------
Constructor   Meaning
-----------   --------------------------------------------------
Rx            Pure rotation about the x-axis
Ry            Pure rotation about the y-axis
Rz            Pure rotation about the z-axis
RPY           specified as roll-pitch-yaw angles
Eul           specified as Euler angles
AngVec        specified as rotational axis and rotation angle
Rand          random rotation
Exp           specified as se(2) or se(3) matrix
empty         no values
-----------   --------------------------------------------------

Non-group operations
^^^^^^^^^^^^^^^^^^^^

The classes ``SE3``, ``SO3``, ``SE2``, ``SO2`` and ``UnitQuaternion`` support vector transformation when 
premultiplying a vector (or a set of vectors columnwise in a NumPy array) using the ``*`` operator.
This is either rotation about the origin (for ``SO3``, ``SO2`` and ``UnitQuaternion``) or rotation and translation (``SE3``, ``SE2``).  
For ``UnitQuaternion`` this is performed directly using Hamilton products :math:`q \circ \mathring{v} \circ q^{-1}`.
For ``SO3`` and ``SO2`` this is a matrix-vector product, for ``SE3`` and ``SE2`` this is a matrix-vector product with the vectors
being first converted to homogeneous form, and the result converted back to Euclidean form.

Scalar multiplication, addition and subtraction are not defined group operations so the result will be a NumPy array rather than a class,
and the operations are performed elementwise, for example::

  >>> T = SE3.Rx(0.3)
  >>> T - T
  array([[0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.]])

or in the case of a scalar broadcast to each element::

  >>> T - 1
  array([[ 0.        , -1.        , -1.        , -1.        ],
        [-1.        , -0.04466351, -1.29552021, -1.        ],
        [-1.        , -0.70447979, -0.04466351, -1.        ],
        [-1.        , -1.        , -1.        ,  0.        ]])
  >>> 2 * T
  array([[ 2.        ,  0.        ,  0.        ,  0.        ],
        [ 0.        ,  1.91067298, -0.59104041,  0.        ],
        [ 0.        ,  0.59104041,  1.91067298,  0.        ],
        [ 0.        ,  0.        ,  0.        ,  2.        ]])

The exception is the ``Quaternion`` class which supports these since a quaternion is a ring not a group::

  >>> q
  1.000000 < 2.000000, 3.000000, 4.000000 >
  >>> 2 * q
  2.000000 < 4.000000, 6.000000, 8.000000 >

Compare this to the unit quaternion case::

  >>> q = UnitQuaternion([1, 2, 3, 4])
  >>> q
  0.182574 << 0.365148, 0.547723, 0.730297 >>

  >>> 2 * q
  0.365148 < 0.730297, 1.095445, 1.460593 >

Noting that unit quaternions are denoted by double angle bracket delimiters of their vector part,
whereas a general quaternion uses single angle brackets.  The product of a general quaternion and a 
unit quaternion is always a general quaternion.


Displaying values
-----------------

Each class has a compact text representation via its *repr* method and its ``str()`` method.
The ``printline()`` methods prints a single-line for tabular listing to the console, file and returns a string::

  >>> _ = X.printline()
  t =      0.6,    -0.29,    -0.98; rpy/zyx =  1.5e+02,       36,      -44 deg

The classes ``SE3``, ``SO3``, ``SE2`` and ``SO2`` can provide colorized text output to the console::

  >>> T = SE3()
  >>> T.print()

.. image:: ../../figs/colored_output.png

with rotational elements in red, translational elements in blue and constants in grey.

Graphics
--------

Each class has a ``plot`` method that displays the corresponding pose as a coordinate frame, for example::

  >>> X = SE3.Rand()
  >>> X.plot()

.. image:: figs/fig1.png

and there are many display options.

The ``animate`` method animates the motion of the coordinate frame from the null-pose, for example:

  >>> X = SE3.Rand()
  >>> X.animate(frame='A', arrow=False)

.. image:: figs/animate.gif


Constructors
------------

The constructor for each class can accept:

* no arguments, in which case the identity element is created::

    >>> SE2()
    SE2(array([[1., 0., 0.],
              [0., 1., 0.],
              [0., 0., 1.]]))

* class specific values, eg. ``SE2(x, y, theta)`` or ``SE3(x, y, z)``, for example::

    >>> SE2(1, 2, 0.3)
    SE2(array([[ 0.95533649, -0.29552021,  1.        ],
              [ 0.29552021,  0.95533649,  2.        ],
              [ 0.        ,  0.        ,  1.        ]]))
    >>> UnitQuaternion([1, 0, 0, 0])
    1.000000 << 0.000000, 0.000000, 0.000000 >>

* a numeric value for the class as a NumPy array or a 1D list or tuple which will be checked for validity::

    >>> SE2(numpy.identity(3))
    SE2(array([[1., 0., 0.],
              [0., 1., 0.],
              [0., 0., 1.]]))

* a list of numeric values, each of which will be checked for validity::

    >>> X = SE2([numpy.identity(3), numpy.identity(3), numpy.identity(3), numpy.identity(3)])
    >>> X
    SE2([
    array([[1., 0., 0.],
          [0., 1., 0.],
          [0., 0., 1.]]),
    array([[1., 0., 0.],
          [0., 1., 0.],
          [0., 0., 1.]]),
    array([[1., 0., 0.],
          [0., 1., 0.],
          [0., 0., 1.]]),
    array([[1., 0., 0.],
          [0., 1., 0.],
          [0., 0., 1.]]) ])
    >>> len(X)
    4

.. _list-powers:

List capability
---------------

Each of these object classes has ``UserList`` as a base class which means it inherits all the functionality of
a Python list

.. code:: python

  >>> R = SO3.Rx(0.3)
  >>> len(R)
     1

.. code:: python

  >>> R = SO3.Rx(np.arange(0, 2*np.pi, 0.2)))
  >>> len(R)
    32
  >> R[0]
     1         0         0          
     0         1         0          
     0         0         1     
  >> R[-1]
     1         0         0          
     0         0.996542  0.0830894  
     0        -0.0830894 0.996542

where each item is an object of the same class as that it was extracted from.
Slice notation is also available, eg. ``R[0:-1:3]`` is a new SO3 instance containing every third element of ``R``.

In particular it includes an iterator allowing comprehensions

.. code:: python

  >>> [x.eul for x in R]
  [array([ 90.        ,   4.76616702, -90.        ]),
   array([ 90.        ,  16.22532292, -90.        ]),
   array([ 90.        ,  27.68447882, -90.        ]),
     .
     .
   array([-90.       ,  11.4591559,  90.       ]),
   array([0., 0., 0.])]


Useful functions that be used on such objects include

=============  ================================================ 
Method              Operation
=============  ================================================ 
``clear``       Clear all elements, object now has zero length
``append``      Append a single element
``del``
``enumerate``   Iterate over the elments
``extend``      Append a list of same type pose objects
``insert``      Insert an element
``len``         Return the number of elements
``map``         Map a function of each element
``pop``         Remove first element and return it
``slice``       Index from a slice object
``zip``         Iterate over the elments
=============  ================================================ 


Vectorization
-------------

For most methods, if applied to an object that contains N elements, the result will be the appropriate return object type with N elements.

Most binary operations (`*`, `*=`, `**`, `+`, `+=`, `-`, `-=`, `==`, `!=`) are vectorized.  For the case::

  Z = X op Y

the lengths of the operands and the results are given by


======   ======   ======  ========================
     operands           results
---------------   --------------------------------
len(X)   len(Y)   len(Z)     results         
======   ======   ======  ========================
  1        1        1       Z    = X op Y
  1        M        M       Z[i] = X op Y[i]
  M        1        M       Z[i] = X[i] op Y
  M        M        M       Z[i] = X[i] op Y[i]
======   ======   ======  ========================

Any other combination of lengths is not allowed and will raise a ``ValueError`` exception.

Implementation
--------------

=========  ===========================
Operator      dunder method
=========  ===========================
  ``*``      **__mul__** , __rmul__
  ``*=``     __imul__
  ``/``      **__truediv__**
  ``/=``     __itruediv__
  ``**``     **__pow__**
  ``**=``    __ipow__
  ``+``      **__add__**, __radd__
  ``+=``     __iadd__
  ``-``      **__sub__**, __rsub__
  ``-=``     __isub__
=========  ===========================

This online documentation includes just the method shown in bold.
The other related methods all invoke that method.

Low-level spatial math
======================

All the classes just described abstract the ``base`` package which represent the spatial-math object as a numpy.ndarray.

The inputs to functions in this package are either floats, lists, tuples or numpy.ndarray objects describing vectors or arrays.  Functions that require a vector can be passed a list, tuple or numpy.ndarray for a vector -- described in the documentation as being of type *array_like*.

Numpy vectors are somewhat different to MATLAB, and is a gnarly aspect of numpy.  Numpy arrays have a shape described by a shape tuple which is a list of the dimensions.  Typically all ``np.ndarray`` vectors have the shape (N,), that is, they have only one dimension.  The ``@`` product of an (M,N) array and a (N,) vector is a (M,) array.  A numpy column vector has shape (N,1) and a row vector has shape (1,N) but functions also accept row (1,N)  and column (N,1) vectors.  
Iterating over a numpy.ndarray is done by row, not columns as in MATLAB.  Iterating over a 1D array (N,) returns consecutive elements, iterating a row vector (1,N) returns the entire row, iterating a column vector (N,1) returns consecutive elements (rows).

For example an SE(2) pose is represented by a 3x3 numpy array, an ndarray with shape=(3,3). A unit quaternion is 
represented by a 4-element numpy array, an ndarray with shape=(4,).

=================    ================   ===================
Spatial object       equivalent class   numpy.ndarray shape
=================    ================   ===================
2D rotation SO(2)    SO2                   (2,2)
2D pose SE(2)        SE2                   (3,3)
3D rotation SO(3)    SO3                   (3,3)
3D poseSE3 SE(3)     SE3                   (3,3)
3D rotation          UnitQuaternion        (4,)
n/a                  Quaternion            (4,)
=================    ================   ===================

Tjhe classes ``SO2``, ```SE2``, ```SO3``, ``SE3``, ``UnitQuaternion`` can operate conveniently on lists but the ``base`` functions do not support this.
If you wish to work with these functions and create lists of pose objects you could keep the numpy arrays in high-order numpy arrays (ie. add an extra dimensions),
or keep them in a list, tuple or any other python contai described in the [high-level spatial math section](#high-level-classes).

Let's show a simple example:

.. code-block:: python
   :linenos:

    >>> import spatialmath.base.transforms as base
    >>> base.rotx(0.3)
    array([[ 1.        ,  0.        ,  0.        ],
           [ 0.        ,  0.95533649, -0.29552021],
           [ 0.        ,  0.29552021,  0.95533649]])

    >>> base.rotx(30, unit='deg')
    array([[ 1.       ,  0.       ,  0.       ],
           [ 0.       ,  0.8660254, -0.5      ],
           [ 0.       ,  0.5      ,  0.8660254]])

    >>> R = base.rotx(0.3) @ base.roty(0.2)

At line 1 we import all the base functions into the namespae ``base``.
In line 12 when we multiply the matrices we need to use the `@` operator to perform matrix multiplication.  The `*` operator performs element-wise multiplication, which is equivalent to the MATLAB ``.*`` operator.

We also support multiple ways of passing vector information to functions that require it:

* as separate positional arguments

.. code:: python

  transl2(1, 2)
  array([[1., 0., 1.],
         [0., 1., 2.],
         [0., 0., 1.]])

* as a list or a tuple

.. code:: python

  transl2( [1,2] )
  array([[1., 0., 1.],
         [0., 1., 2.],
         [0., 0., 1.]])

  transl2( (1,2) )
  array([[1., 0., 1.],
         [0., 1., 2.],
         [0., 0., 1.]])


* or as a `numpy` array

.. code:: python

  transl2( np.array([1,2]) )
  array([[1., 0., 1.],
         [0., 1., 2.],
         [0., 0., 1.]])


There is a single module that deals with quaternions, regular quaternions and unit quaternions, and the representation is a `numpy` array of four elements.  As above, functions can accept the `numpy` array, a list, dict or `numpy` row or column vectors.


.. code:: python

  >>> import spatialmath.base.quaternion as quat
  >>> q = quat.qqmul([1,2,3,4], [5,6,7,8])
  >>> q
  array([-60,  12,  30,  24])
  >>> quat.qprint(q)
  -60.000000 < 12.000000, 30.000000, 24.000000 >
  >>> quat.qnorm(q)
  72.24956747275377

Functions exist to convert to and from SO(3) rotation matrices and a 3-vector representation.  The latter is often used for SLAM and bundle adjustment applications, being a minimal representation of orientation.

Graphics
--------

If ``matplotlib`` is installed then we can add 2D coordinate frames to a figure in a variety of styles:

.. code-block:: python
   :linenos:

    trplot2( transl2(1,2), frame='A', rviz=True, width=1)
    trplot2( transl2(3,1), color='red', arrow=True, width=3, frame='B')
    trplot2( transl2(4, 3)@trot2(math.pi/3), color='green', frame='c')
    plt.grid(True)

.. figure:: ./figs/transforms2d.png 
   :align: center

   Output of ``trplot2``

If a figure does not yet exist one is added.  If a figure exists but there is no 2D axes then one is added.  To add to an existing axes you can pass this in using the ``axes`` argument.  By default the frames are drawn with lines or arrows of unit length.  Autoscaling is enabled.

Similarly, we can plot 3D coordinate frames in a variety of styles:

.. code-block:: python
   :linenos:

    trplot( transl(1,2,3), frame='A', rviz=True, width=1, dims=[0, 10, 0, 10, 0, 10])
    trplot( transl(3,1, 2), color='red', width=3, frame='B')
    trplot( transl(4, 3, 1)@trotx(math.pi/3), color='green', frame='c', dims=[0,4,0,4,0,4])

.. figure:: ./figs/transforms3d.png
   :align: center

   Output of ``trplot``

The ``dims`` option in lines 1 and 3 sets the workspace dimensions.  Note that the last set value is what is displayed.

Depending on the backend you are using you may need to include

.. code-block:: python

    plt.show()


Symbolic support
----------------

Some functions have support for symbolic variables, for example

.. code:: python

  import sympy

  theta = sym.symbols('theta')
  print(rotx(theta))
  [[1 0 0]
   [0 cos(theta) -sin(theta)]
   [0 sin(theta) cos(theta)]]

The resulting `numpy` array is an array of symbolic objects not numbers &ndash; the constants are also symbolic objects.  You can read the elements of the matrix

.. code:: python

  >>> a = T[0,0]
  >>> a
    1
  >>> type(a)
   int

  >>> a = T[1,1]
  >>> a 
  cos(theta)
  >>> type(a)
   cos

We see that the symbolic constants are converted back to Python numeric types on read.

Similarly when we assign an element or slice of the symbolic matrix to a numeric value, they are converted to symbolic constants on the way in.

.. code:: python

  >>> T[0,3] = 22
  >>> print(T)
  [[1 0 0 22]
   [0 cos(theta) -sin(theta) 0]
   [0 sin(theta) cos(theta) 0]
   [0 0 0 1]]

but you can't write a symbolic value into a floating point matrix

.. code:: python

  >>> T = trotx(0.2)

  >>> T[0,3]=theta
  Traceback (most recent call last):
    .
    .
  TypeError: can't convert expression to float

MATLAB compatability
--------------------

We can create a MATLAB like environment by

.. code-block:: python

    from spatialmath  import *
    from spatialmath.base  import *

which has familiar functions like ``rotx`` and ``rpy2r`` available, as well as classes like ``SE3``

.. code-block:: python

  R = rotx(0.3)
  R2 = rpy2r(0.1, 0.2, 0.3)

  T = SE3(1, 2, 3)

.. |reg|    unicode:: U+000AE .. REGISTERED SIGN


