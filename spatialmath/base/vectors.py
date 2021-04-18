# Part of Spatial Math Toolbox for Python
# Copyright (c) 2000 Peter Corke
# MIT Licence, see details in top-level file: LICENCE

"""
Functions to manipulate vectors

Vector arguments are what numpy refers to as ``array_like`` and can be a list,
tuple, numpy array, numpy row vector or numpy column vector.
"""

# pylint: disable=invalid-name

import math
import numpy as np
from spatialmath.base import getvector

try:  # pragma: no cover
    # print('Using SymPy')
    import sympy

    _symbolics = True

except ImportError:  # pragma: no cover
    _symbolics = False

_eps = np.finfo(np.float64).eps


def colvec(v):
    """
    Create a column vector

    :param v: any vector
    :type v: array_like(n)
    :return: a column vector
    :rtype: ndarray(n,1)

    Convert input to a column vector.

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> colvec([1, 2, 3])
    """
    v = getvector(v)
    return np.array(v).reshape((len(v), 1))


def unitvec(v):
    """
    Create a unit vector

    :param v: any vector
    :type v: array_like(n)
    :return: a unit-vector parallel to ``v``.
    :rtype: ndarray(n)
    :raises ValueError: for zero length vector

    ``unitvec(v)`` is a vector parallel to `v` of unit length.

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> unitvec([3, 4])

    :seealso: :func:`~numpy.linalg.norm`

    """

    v = getvector(v)
    n = norm(v)

    if n > 100 * _eps:  # if greater than eps
        return v / n
    else:
        return None


def unitvec_norm(v):
    """
    Create a unit vector

    :param v: any vector
    :type v: array_like(n)
    :return: a unit-vector parallel to ``v`` and the norm
    :rtype: (ndarray(n), float)
    :raises ValueError: for zero length vector

    ``unitvec(v)`` is a vector parallel to `v` of unit length.

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> unitvec([3, 4])

    :seealso: :func:`~numpy.linalg.norm`

    """

    v = getvector(v)
    n = np.linalg.norm(v)

    if n > 100 * _eps:  # if greater than eps
        return (v / n, n)
    else:
        return None, None

def norm(v):
    """
    Norm of vector

    :param v: any vector
    :type v: array_like(n)
    :return: norm of vector
    :rtype: float

    ``norm(v)`` is the 2-norm (length or magnitude) of the vector ``v``.

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> norm([3, 4])

    .. note:: This function does not use NumPy, it is ~2x faster than 
        `numpy.linalg.norm()` for a 3-vector

    :seealso: :func:`~spatialmath.base.unit`

    :SymPy: supported
    """
    sum = 0
    for x in v:
        sum += x * x

    if _symbolics and isinstance(sum, sympy.Expr):
        return sympy.sqrt(sum)
    else:
        return math.sqrt(sum)

def normsq(v):
    """
    Squared norm of vector

    :param v: any vector
    :type v: array_like(n)
    :return: norm of vector
    :rtype: float

    ``norm(sq)`` is the sum of squared elements of the vector ``v`` 
    or :math:`|v|^2`.

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> normsq([2, 3])

    .. note:: This function does not use NumPy, it is ~2x faster than 
        `numpy.linalg.norm() ** 2` for a 3-vector

    :seealso: :func:`~spatialmath.base.unit`

    :SymPy: supported
    """
    sum = 0
    for x in v:
        sum += x * x

    return sum

def cross(u, v):
    """
    Cross product of vectors

    :param u: any vector
    :type u: array_like(3)
    :param v: any vector
    :type v: array_like(3)
    :return: cross product
    :rtype: nd.array(3)

    ``cross(u, v)`` is the cross product of the vectors ``u`` and ``v``.

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> cross([1, 0, 0], [0, 1, 0])

    .. note:: This function does not use NumPy, it is ~1.5x faster than 
        `numpy.cross()`

    :seealso: :func:`~spatialmath.base.unit`

    :SymPy: supported
    """
    return np.r_[
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0]
                ]

def isunitvec(v, tol=10):
    """
    Test if vector has unit length

    :param v: vector to test
    :type v: ndarray(n)
    :param tol: tolerance in units of eps
    :type tol: float
    :return: whether vector has unit length
    :rtype: bool

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> isunitvec([1, 0])
        >>> isunitvec([1, 2])

    :seealso: unit, iszerovec, isunittwist
    """
    return abs(np.linalg.norm(v) - 1) < tol * _eps


def iszerovec(v, tol=10):
    """
    Test if vector has zero length

    :param v: vector to test
    :type v: ndarray(n)
    :param tol: tolerance in units of eps
    :type tol: float
    :return: whether vector has zero length
    :rtype: bool

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> iszerovec([0, 0])
        >>> iszerovec([1, 2])

    :seealso: unit, isunitvec, isunittwist
    """
    return np.linalg.norm(v) < tol * _eps

def iszero(v, tol=10):
    """
    Test if scalar is zero

    :param v: value to test
    :type v: float
    :param tol: tolerance in units of eps
    :type tol: float
    :return: whether value is zero
    :rtype: bool

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> iszero(0)
        >>> iszero(1)

    :seealso: unit, iszerovec, isunittwist
    """
    return abs(v) < tol * _eps

def isunittwist(v, tol=10):
    r"""
    Test if vector represents a unit twist in SE(2) or SE(3)

    :param v: twist vector to test
    :type v: array_like(6)
    :param tol: tolerance in units of eps
    :type tol: float
    :return: whether twist has unit length
    :rtype: bool
    :raises ValueError: for incorrect vector length


    Vector is is intepretted as :math:`[v, \omega]` where :math:`v \in \mathbb{R}^n` and
    :math:`\omega \in \mathbb{R}^1` for SE(2) and :math:`\omega \in \mathbb{R}^3` for SE(3).

    A unit twist can be a:

    - unit rotational twist where :math:`|| \omega || = 1`, or
    - unit translational twist where :math:`|| \omega || = 0` and :math:`|| v || = 1`.

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> isunittwist([1, 2, 3, 1, 0, 0])
        >>> isunittwist([0, 0, 0, 2, 0, 0])

    :seealso: unit, isunitvec
    """
    v = getvector(v)

    if len(v) == 6:
        # test for SE(3) twist
        return isunitvec(v[3:6], tol=tol) or (np.linalg.norm(v[3:6]) < tol * _eps and isunitvec(v[0:3], tol=tol))
    else:
        raise ValueError


def isunittwist2(v, tol=10):
    r"""
    Test if vector represents a unit twist in SE(2) or SE(3)

    :param v: twist vector to test
    :type v: array_like(3)
    :param tol: tolerance in units of eps
    :type tol: float
    :return: whether vector has unit length
    :rtype: bool
    :raises ValueError: for incorrect vector length

    Vector is is intepretted as :math:`[v, \omega]` where :math:`v \in \mathbb{R}^n` and
    :math:`\omega \in \mathbb{R}^1` for SE(2) and :math:`\omega \in \mathbb{R}^3` for SE(3).

    A unit twist can be a:

    - unit rotational twist where :math:`|| \omega || = 1`, or
    - unit translational twist where :math:`|| \omega || = 0` and :math:`|| v || = 1`.

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> isunittwist2([1, 2, 1])
        >>> isunittwist2([0, 0, 2])

    :seealso: unit, isunitvec
    """
    v = getvector(v)

    if len(v) == 3:
        # test for SE(2) twist
        return isunitvec(v[2], tol=tol) or (np.abs(v[2]) < tol * _eps and isunitvec(v[0:2], tol=tol))
    else:
        raise ValueError


def unittwist(S, tol=10):
    """
    Convert twist to unit twist

    :param S: twist vector
    :type S: array_like(6)
    :param tol: tolerance in units of eps
    :type tol: float
    :return: unit twist
    :rtype: ndarray(6)

    A unit twist is a twist where:

    - the rotation part has unit magnitude
    - if the rotational part is zero, then the translational part has unit magnitude

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> unittwist([2, 4, 6, 2, 0, 0])
        >>> unittwist([2, 0, 0, 0, 0, 0])

    Returns None if the twist has zero magnitude
    """

    S = getvector(S, 6)

    if iszerovec(S, tol=tol):
        return None

    v = S[0:3]
    w = S[3:6]

    if iszerovec(w):
        th = norm(v)
    else:
        th = norm(w)

    return S / th


def unittwist_norm(S, tol=10):
    """
    Convert twist to unit twist and norm

    :param S: twist vector
    :type S: array_like(6)
    :param tol: tolerance in units of eps
    :type tol: float
    :return: unit twist and scalar motion
    :rtype: tuple (ndarray(6), float)

    A unit twist is a twist where:

    - the rotation part has unit magnitude
    - if the rotational part is zero, then the translational part has unit magnitude

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> S, n = unittwist_norm([1, 2, 3, 1, 0, 0])
        >>> print(S, n)
        >>> S, n = unittwist_norm([0, 0, 0, 2, 0, 0])
        >>> print(S, n)
        >>> S, n = unittwist_norm([0, 0, 0, 0, 0, 0])
        >>> print(S, n)

    .. note:: Returns (None,None) if the twist has zero magnitude
    """

    S = getvector(S, 6)

    if iszerovec(S, tol=tol):
        return (None, None)

    v = S[0:3]
    w = S[3:6]

    if iszerovec(w):
        th = norm(v)
    else:
        th = norm(w)

    return (S / th, th)


def unittwist2(S):
    """
    Convert twist to unit twist

    :param S: twist vector
    :type S: array_like(3)
    :return: unit twist 
    :rtype: ndarray(3)

    A unit twist is a twist where:

    - the rotation part has unit magnitude
    - if the rotational part is zero, then the translational part has unit magnitude

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> unittwist2([2, 4, 2)
        >>> unittwist2([2, 0, 0])

    """

    S = getvector(S, 3)
    v = S[0:2]
    w = S[2]

    if iszero(w):
        th = norm(v)
    else:
        th = abs(w)

    return S / th

def unittwist2_norm(S):
    """
    Convert twist to unit twist

    :param S: twist vector
    :type S: array_like(3)
    :return: unit twist and scalar motion
    :rtype: tuple (ndarray(3), float)

    A unit twist is a twist where:

    - the rotation part has unit magnitude
    - if the rotational part is zero, then the translational part has unit magnitude

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> unittwist2([2, 4, 2)
        >>> unittwist2([2, 0, 0])

    """

    S = getvector(S, 3)
    v = S[0:2]
    w = S[2]

    if iszero(w):
        th = norm(v)
    else:
        th = abs(w)

    return (S / th, th)

def angdiff(a, b=None):
    """
    Angular difference

    :param a: angle in radians
    :type a: scalar or array_like
    :param b: angle in radians
    :type b: scalar or array_like
    :return: angular difference a-b
    :rtype: scalar or array_like

    - ``angdiff(a, b)`` is the difference ``a - b`` wrapped to the range
      :math:`[-\pi, \pi)`.  This is the operator :math:`a \circleddash b` used
      in the RVC book
        - If ``a`` and ``b`` are both scalars, the result is scalar
        - If ``a`` is array_like, the result is a NumPy array ``a[i]-b``
        - If ``a`` is array_like, the result is a NumPy array ``a-b[i]``
        - If ``a`` and ``b`` are both vectors of the same length, the result is
          a NumPy array ``a[i]-b[i]``

    - ``angdiff(a)`` is the angle or vector of angles ``a`` wrapped to the range
      :math:`[-\pi, \pi)`.
        - If ``a`` is a scalar, the result is scalar
        - If ``a`` is array_like, the result is a NumPy array

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> from math import pi
        >>> angdiff(0, 2 * pi)
        >>> angdiff(0.9 * pi, -0.9 * pi) / pi
        >>> angdiff(3 * pi)

    """
    if b is None:
        return np.mod(a + math.pi, 2 * math.pi) - math.pi
    else:
        return np.mod(a - b + math.pi, 2 * math.pi) - math.pi

def removesmall(v, tol=100):
    """
    Set small values to zero

    :param v: any vector
    :type v: array_like(n) or ndarray(n,m)
    :param tol: Tolerance in units of eps, defaults to 100
    :type tol: int, optional
    :return: vector with small values set to zero
    :rtype: ndarray(n) or ndarray(n,m)

    Values with absolute value less than ``tol`` will be set to zero.

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> a = np.r_[1, 2, 3, 1e-16]
        >>> print(a)
        >>> a = removesmall(a)
        >>> print(a)
        >>> print(a[3])

    """
    return np.where(abs(v) < tol * _eps, 0, v)


if __name__ == '__main__':  # pragma: no cover
    import pathlib

    exec(open(pathlib.Path(__file__).parent.parent.parent.absolute() / "tests" / "base" / "test_vectors.py").read())  # pylint: disable=exec-used
