"""
This modules contains functions to create and transform rotation matrices
and homogeneous tranformation matrices.

Vector arguments are what numpy refers to as ``array_like`` and can be a list,
tuple, numpy array, numpy row vector or numpy column vector.

"""

# This file is part of the SpatialMath toolbox for Python
# https://github.com/petercorke/spatialmath-python
#
# MIT License
#
# Copyright (c) 1993-2020 Peter Corke
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Contributors:
#
#     1. Luis Fernando Lara Tobar and Peter Corke, 2008
#     2. Josh Carrigg Hodson, Aditya Dua, Chee Ho Chan, 2017 (robopy)
#     3. Peter Corke, 2020

# pylint: disable=invalid-name

import math
import numpy as np
from spatialmath.base import argcheck


_eps = np.finfo(np.float64).eps


def colvec(v):
    """
    Create a column vector

    :param v: an N-vector
    :type v: array like
    :return: a column vector
    :rtype: NumPy ndarray, shape=(N,1)
    """
    return np.array(v).reshape((len(v), 1))


# ---------------------------------------------------------------------------------------#
def unitvec(v):
    """
    Create a unit vector

    :param v: n-dimensional vector
    :type v: array_like
    :return: a unit-vector parallel to V.
    :rtype: numpy.ndarray
    :raises ValueError: for zero length vector

    ``unitvec(v)`` is a vector parallel to `v` of unit length.

    :seealso: norm

    """

    v = argcheck.getvector(v)
    n = np.linalg.norm(v)

    if n > 100 * _eps:  # if greater than eps
        return v / n
    else:
        return None


def norm(v):
    """
    Norm of vector

    :param v: n-vector as a list, dict, or a numpy array, row or column vector
    :return: norm of vector
    :rtype: float

    ``norm(v)`` is the 2-norm (length or magnitude) of the vector ``v``.

    :seealso: unit

    """
    return np.linalg.norm(v)


def isunitvec(v, tol=10):
    """
    Test if vector has unit length

    :param v: vector to test
    :type v: numpy.ndarray
    :param tol: tolerance in units of eps
    :type tol: float
    :return: whether vector has unit length
    :rtype: bool

    :seealso: unit, isunittwist
    """
    return abs(np.linalg.norm(v) - 1) < tol * _eps


def iszerovec(v, tol=10):
    """
    Test if vector has zero length

    :param v: vector to test
    :type v: numpy.ndarray
    :param tol: tolerance in units of eps
    :type tol: float
    :return: whether vector has zero length
    :rtype: bool

    :seealso: unit, isunittwist
    """
    return np.linalg.norm(v) < tol * _eps


def isunittwist(v, tol=10):
    r"""
    Test if vector represents a unit twist in SE(2) or SE(3)

    :param v: vector to test
    :type v: array_like
    :param tol: tolerance in units of eps
    :type tol: float
    :return: whether vector has unit length
    :rtype: bool

    Vector is is intepretted as :math:`[v, \omega]` where :math:`v \in \mathbb{R}^n` and
    :math:`\omega \in \mathbb{R}^1` for SE(2) and :math:`\omega \in \mathbb{R}^3` for SE(3).

    A unit twist can be a:

    - unit rotational twist where :math:`|| \omega || = 1`, or
    - unit translational twist where :math:`|| \omega || = 0` and :math:`|| v || = 1`.

    :seealso: unit, isunitvec
    """
    v = argcheck.getvector(v)

    if len(v) == 6:
        # test for SE(3) twist
        return isunitvec(v[3:6], tol=tol) or (np.linalg.norm(v[3:6]) < tol * _eps and isunitvec(v[0:3], tol=tol))
    elif len(v) == 3:
        return isunitvec(v[2], tol=tol) or (abs(v[2]) < tol * _eps and isunitvec(v[0:2], tol=tol))
    else:
        raise ValueError


def isunittwist2(v, tol=10):
    r"""
    Test if vector represents a unit twist in SE(2) or SE(3)

    :param v: vector to test
    :type v: array_like
    :param tol: tolerance in units of eps
    :type tol: float
    :return: whether vector has unit length
    :rtype: bool

    Vector is is intepretted as :math:`[v, \omega]` where :math:`v \in \mathbb{R}^n` and
    :math:`\omega \in \mathbb{R}^1` for SE(2) and :math:`\omega \in \mathbb{R}^3` for SE(3).

    A unit twist can be a:

    - unit rotational twist where :math:`|| \omega || = 1`, or
    - unit translational twist where :math:`|| \omega || = 0` and :math:`|| v || = 1`.

    :seealso: unit, isunitvec
    """
    v = argcheck.getvector(v)

    if len(v) == 3:
        # test for SE(2) twist
        return isunitvec(v[2], tol=tol) or (np.abs(v[2]) < tol * _eps and isunitvec(v[0:2], tol=tol))
    else:
        raise ValueError


def unittwist(S, tol=10):
    """
    Convert twist to unit twist

    :param S: twist as a 6-vector
    :type S: array_like
    :param tol: tolerance in units of eps
    :type tol: float
    :return: unit twist and scalar motion
    :rtype: np.ndarray, shape=(6,)

    A unit twist is a twist where:

    - the rotation part has unit magnitude
    - if the rotational part is zero, then the translational part has unit magnitude

    Returns None if the twist has zero magnitude
    """

    s = argcheck.getvector(S, 6)

    if iszerovec(s, tol=tol):
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

    :param S: twist as a 6-vector
    :type S: array_like
    :param tol: tolerance in units of eps
    :type tol: float
    :return: unit twist and scalar motion
    :rtype: tuple (np.ndarray shape=(6,), theta)

    A unit twist is a twist where:

    - the rotation part has unit magnitude
    - if the rotational part is zero, then the translational part has unit magnitude

    Returns (None,None) if the twist has zero magnitude
    """

    s = argcheck.getvector(S, 6)

    if iszerovec(s, tol=tol):
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

    :param S: twist as a 3-vector
    :type S: array_like
    :return: unit twist and scalar motion
    :rtype: tuple (unit_twist, theta)

    A unit twist is a twist where:

    - the rotation part has unit magnitude
    - if the rotational part is zero, then the translational part has unit magnitude
    """

    S = argcheck.getvector(S, 3)
    v = S[0:2]
    w = S[2]

    if iszerovec(w):
        th = norm(v)
    else:
        th = norm(w)

    return (S / th, th)


def angdiff(a, b):
    """
    Angular difference

    :param a: angle in radians
    :type a: scalar or array_like
    :param b: angle in radians
    :type b: scalar or array_like
    :return: angular difference a-b
    :rtype: scalar or array_like

    - If ``a`` and ``b`` are both scalars, the result is scalar
    - If ``a`` is array_like, the result is a vector a[i]-b
    - If ``a`` is array_like, the result is a vector a-b[i]
    - If ``a`` and ``b`` are both vectors of the same length, the result is a vector a[i]-b[i]
    """

    return np.mod(a - b + math.pi, 2 * math.pi) - math.pi

def removesmall(v, tol=100):
    """
    Set small values to zero

    :param v: Input vector
    :type v: array-like
    :param tol: Tolerance in units of eps, defaults to 100
    :type tol: int, optional
    :return: Input vector with small values set to zero
    :rtype: NumPy ndarray

    Values with absolute value less than ``tol`` will be set to zero.
    """
    return np.where(abs(v) < tol * _eps, 0, v)


if __name__ == '__main__':  # pragma: no cover
    import pathlib

    exec(open(pathlib.Path(__file__).parent.absolute() / "test_argcheck.py").read())  # pylint: disable=exec-used
