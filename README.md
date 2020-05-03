[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)[![Build Status](https://travis-ci.com/petercorke/spatialmath-python.svg?branch=master)](https://travis-ci.com/petercorke/spatialmath-python)
![Coverage](https://codecov.io/gh/petercorke/spatialmath-python/branch/master/graph/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/petercorke/spatialmath-python/graphs/commit-activity)
[![GitHub stars](https://img.shields.io/github/stars/petercorke/spatialmath-python.svg?style=social&label=Star&maxAge=2592000)](https://GitHub.com/petercorke/spatialmath-python/stargazers/)

# Spatial Maths for Python

This is a Python implementation of the [Spatial Math Toolbox for MATLAB<sup>&reg;</sup>](https://github.com/petercorke/spatial-math), which is a standalone component of the [Robotics Toolbox for MATLAB<sup>&reg;</sup>](https://github.com/petercorke/robotics-toolbox-matlab).

Spatial mathematics capability underpins all of robotics and robotic vision where we need to describe the position, orientation or pose of objects in 2D or 3D spaces.


# Examples
## Low-level spatial math

Import the low-level transform functions

```
>>> import spatialmath.base as tr
```

We can create a 3D rotation matrix

```
>>> tr.rotx(0.3)
array([[ 1.        ,  0.        ,  0.        ],
       [ 0.        ,  0.95533649, -0.29552021],
       [ 0.        ,  0.29552021,  0.95533649]])

>>> tr.rotx(30, unit='deg')
array([[ 1.       ,  0.       ,  0.       ],
       [ 0.       ,  0.8660254, -0.5      ],
       [ 0.       ,  0.5      ,  0.8660254]])
```
The results are `numpy` arrays so to perform matrix multiplication you need to use the `@` operator, for example

```
rotx(0.3) @ roty(0.2)
```

We also support multiple ways of passing vector information to functions that require it:

* as separate positional arguments

```
transl2(1, 2)
array([[1., 0., 1.],
       [0., 1., 2.],
       [0., 0., 1.]])
```

* as a list or a tuple

```
transl2( [1,2] )
array([[1., 0., 1.],
       [0., 1., 2.],
       [0., 0., 1.]])

transl2( (1,2) )
Out[444]: 
array([[1., 0., 1.],
       [0., 1., 2.],
       [0., 0., 1.]])
```

* or as a `numpy` array

```
transl2( np.array([1,2]) )
Out[445]: 
array([[1., 0., 1.],
       [0., 1., 2.],
       [0., 0., 1.]])
```

trplot example
packages, animation

There is a single module that deals with quaternions, unit or not, and the representation is a `numpy` array of four elements.  As above, functions can accept the `numpy` array, a list, dict or `numpy` row or column vectors.

```
>>> from spatialmath.base.quaternion import *
>>> q = qqmul([1,2,3,4], [5,6,7,8])
>>> q
array([-60,  12,  30,  24])
>>> qprint(q)
-60.000000 < 12.000000, 30.000000, 24.000000 >
>>> qnorm(q)
72.24956747275377
```

## Symbolic support

Some functions have support for symbolic variables, for example

```
import sympy

theta = sym.symbols('theta')
print(rotx(theta))
[[1 0 0]
 [0 cos(theta) -sin(theta)]
 [0 sin(theta) cos(theta)]]
```

The resulting `numpy` array is an array of symbolic objects not numbers &ndash; the constants are also symbolic objects.  You can read the elements of the matrix

```
a = T[0,0]

a
Out[258]: 1

type(a)
Out[259]: int

a = T[1,1]
a
Out[256]: 
cos(theta)
type(a)
Out[255]: cos
```
We see that the symbolic constants are converted back to Python numeric types on read.

Similarly when we assign an element or slice of the symbolic matrix to a numeric value, they are converted to symbolic constants on the way in.



## High-level classes

These classes abstract the low-level numpy arrays into objects that obey the rules associated with the mathematical groups SO(2), SE(2), SO(3), SE(3) as well as twists and quaternions.  pose classes `SO2`, `SE2`, `SO3`, `SE3`.

Using classes ensures type safety, for example it stops us mixing a 2D homogeneous transformation with a 3D rotation matrix -- both are 3x3 matrices.

```
>>> import spatialmath as sm
>>> sm.SO3(0.3)
[[ 0.99500417 -0.09983342]
 [ 0.09983342  0.99500417]]
```
or
```
>>> R = sm.SO3.rpy(10, 20, 30, 'deg')
```

These classes are all derived from two parent classes:

* `SuperPose` which provides common functionality for all
* `UserList` which provdides the ability to act like a list 

The latter is important because frequenetly in robotics we want a sequence, a trajectory, of rotation matrices or poses.  However a list of these items has the type `list` and the elements are not enforced to be homogeneous, ie. a list could contain a mixture of classes.

Another option would be to create a `numpy` array of these objects, the upside being it could be a multi-dimensional array.  The downside is that again the array is not guaranteed to be homogeneous.


The approach adopted here is to give these classes list superpowers.  Using the example of SE(3) but applicable to all

```
T = transl(1,2,3) # create a 4x4 np.array

a = SE3(T)
a.append(a)  # append a copy
a.append(a)  # append a copy
type(a)
len(a)
a[1]  # extract one element of the list
for x in a:
  # do a thing
```




```
T[0,3] = 22
print(T)
[[1 0 0 22]
 [0 cos(theta) -sin(theta) 0]
 [0 sin(theta) cos(theta) 0]
 [0 0 0 1]]
```
but you can't write a symbolic value into a floating point matrix

```
T=trotx(0.2)

T[0,3]=theta
Traceback (most recent call last):

  File "<ipython-input-248-b6823f58f38d>", line 1, in <module>
    T[0,3]=th

  File "/opt/anaconda3/lib/python3.7/site-packages/sympy/core/expr.py", line 325, in __float__
    raise TypeError("can't convert expression to float")

TypeError: can't convert expression to float
```

| Function | Symbolic support |
|----------|------------------|
| rot2 | yes |
| transl2 | yes |
| rotx | yes |
| roty | yes |
| rotz | yes |
| transl | yes |
| r2t | yes |
| t2r | yes |
| rotx | yes |
| rotx | yes |







