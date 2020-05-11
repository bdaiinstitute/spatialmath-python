""" 
This modules contains functions to create and transform rotation matrices
and homogeneous tranformation matrices.

Vector arguments are what numpy refers to as ``array_like`` and can be a list,
tuple, numpy array, numpy row vector or numpy column vector.

Versions:

    1. Luis Fernando Lara Tobar and Peter Corke, 2008
    2. Josh Carrigg Hodson, Aditya Dua, Chee Ho Chan, 2017
    3. Peter Corke, 2020
"""

import sys
import math
import numpy as np
from spatialmath.base import argcheck


    
_eps = np.finfo(np.float64).eps

def colvec(v):
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
    
    if n > 100*_eps: # if greater than eps
        return v / n
    else:
        raise ValueError("Vector has zero norm")

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
    return abs(np.linalg.norm(v)-1) < tol*_eps

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
    return np.linalg.norm(v) < tol*_eps


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
    
    if len(v) ==  6:
        # test for SE(3) twist
        return isunitvec(v[3:6], tol=tol) or (np.linalg.norm(v[3:6]) < tol*_eps and isunitvec(v[0:3], tol=tol))
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
        return isunitvec(v[2], tol=tol) or (np.abs(v[2]) < tol*_eps and isunitvec(v[0:2], tol=tol))
    else:
        raise ValueError
        

def unittwist(S):
    """
    Convert twist to unit twist
    
    :param S: twist as a 6-vector
    :type S: array_like
    :return: unit twist and scalar motion
    :rtype: tuple (unit_twist, theta)

    A unit twist is a twist where:
        
    - the rotation part has unit magnitude
    - if the rotational part is zero, then the translational part has unit magnitude
    """
    
    s = argcheck.getvector(S, 6)
    v = S[0:3]
    w = S[3:6]
    
    if iszerovec(w):
        th = norm(v);
    else:
       th = norm(w);

    return (S/th, th)

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
    
    s = argcheck.getvector(S, 3)
    v = S[0:2]
    w = S[2]
    
    if iszerovec(w):
        th = norm(v);
    else:
       th = norm(w);

    return (S/th, th)

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


    
if __name__ == '__main__':  # pragma: no cover
    import pathlib
    import os.path
    
    exec(open(os.path.join(pathlib.Path(__file__).parent.absolute(), "test_transforms.py")).read() )
    
    



