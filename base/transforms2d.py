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
import spatialmath.base.argcheck as argcheck
from spatialmath.base.vectors import *
from spatialmath.base.transformsNd import *


try:
    print('Using SymPy')
    import sympy as sym
    def issymbol(x):
        return isinstance(x, sym.Symbol)
except:
    def issymbol(x):
        return False
    
_eps = np.finfo(np.float64).eps

def colvec(v):
    return np.array(v).reshape((len(v), 1))

# ---------------------------------------------------------------------------------------#
    
def _cos(theta):
    if issymbol(theta):
        return sym.cos(theta)
    else:
        return math.cos(theta)
        
def _sin(theta):
    if issymbol(theta):
        return sym.sin(theta)
    else:
        return math.sin(theta)


# ---------------------------------------------------------------------------------------#
def rot2(theta, unit='rad'):
    """
    Create SO(2) rotation

    :param theta: rotation angle
    :type theta: float
    :param unit: angular units: 'rad' [default], or 'deg'
    :type unit: str
    :return: 2x2 rotation matrix
    :rtype: numpy.ndarray, shape=(2,2)

    - ``ROT2(THETA)`` is an SO(2) rotation matrix (2x2) representing a rotation of THETA radians.
    - ``ROT2(THETA, 'deg')`` as above but THETA is in degrees.
    """
    theta = argcheck.getunit(theta, unit)
    ct = _cos(theta)
    st = _sin(theta)
    R = np.array([
            [ct, -st], 
            [st, ct]  ])
    if not isinstance(theta, sym.Symbol):
        R = R.round(15)
    return R


# ---------------------------------------------------------------------------------------#
def trot2(theta, unit='rad', t=None):
    """
    Create SE(2) pure rotation 

    :param theta: rotation angle about X-axis
    :type theta: float
    :param unit: angular units: 'rad' [default], or 'deg'
    :type unit: str
    :param t: translation 2-vector, defaults to [0,0]
    :type t: array_like    :return: 3x3 homogeneous transformation matrix
    :rtype: numpy.ndarray, shape=(3,3)

    - ``TROT2(THETA)`` is a homogeneous transformation (3x3) representing a rotation of
      THETA radians.
    - ``TROT2(THETA, 'deg')`` as above but THETA is in degrees.
    
    Notes:
    - Translational component is zero.
    """
    T  = np.pad( rot2(theta, unit), (0,1) )
    if t is not None:
        T[:2,2] = argcheck.getvector(t, 2, 'array')
    T[2,2] = 1.0
    return T


# ---------------------------------------------------------------------------------------#
def transl2(x, y=None):
    """
    Create SE(2) pure translation, or extract translation from SE(2) matrix

    :param x: translation along X-axis
    :type x: float
    :param y: translation along Y-axis
    :type y: float
    :return: homogeneous transform matrix or the translation elements of a homogeneous transform
    :rtype: numpy.ndarray, shape=(3,3)

    Create a translational SE(2) matrix:

    - ``T = transl2([X, Y])`` is an SE(2) homogeneous transform (3x3) representing a
      pure translation.
    - ``T = transl2( V )`` as above but the translation is given by a 2-element
      list, dict, or a numpy array, row or column vector.


    Extract the translational part of an SE(2) matrix:

    P = TRANSL2(T) is the translational part of a homogeneous transform as a
    2-element numpy array.  
    """

    if np.isscalar(x):
        T = np.identity(3)
        T[:2,2] = [x, y]
        return T
    elif argcheck.isvector(x, 2):
        T = np.identity(3)
        T[:2,2] = argcheck.getvector(x, 2)
        return T
    elif argcheck.ismatrix(x, (3,3)):
        return x[:2,2]
    else:
        ValueError('bad argument')




def ishom2(T, check=False):
    """
    Test if matrix belongs to SE(2)
    
    :param T: matrix to test
    :type T: numpy.ndarray
    :param check: check validity of rotation submatrix
    :type check: bool
    :return: whether matrix is an SE(2) homogeneous transformation matrix
    :rtype: bool
    
    - ``ISHOM2(T)`` is True if the argument ``T`` is of dimension 3x3
    - ``ISHOM2(T, check=True)`` as above, but also checks orthogonality of the rotation sub-matrix and 
      validitity of the bottom row.
    
    :seealso: isR, isrot2, ishom, isvec
    """
    return T.shape == (3,3) and (not check or (isR(T[:2,:2]) and np.all(T[2,:] == np.array([0,0,1]))))


def isrot2(R, check=False):
    """
    Test if matrix belongs to SO(2)
    
    :param R: matrix to test
    :type R: numpy.ndarray
    :param check: check validity of rotation submatrix
    :type check: bool
    :return: whether matrix is an SO(2) rotation matrix
    :rtype: bool
    
    - ``ISROT(R)`` is True if the argument ``R`` is of dimension 2x2
    - ``ISROT(R, check=True)`` as above, but also checks orthogonality of the rotation matrix.
    
    :seealso: isR, ishom2, isrot
    """
    return R.shape == (2,2) and (not check or isR(R))



# ---------------------------------------------------------------------------------------#
def trexp2(S, theta=None):
    """
    Exponential of so(2) or se(2) matrix

    :param S: so(2), se(2) matrix or equivalent velctor
    :type T: numpy.ndarray, shape=(2,2) or (3,3); array_like
    :param theta: motion
    :type theta: float
    :return: 2x2 or 3x3 matrix exponential in SO(2) or SE(2)
    :rtype: numpy.ndarray, shape=(2,2) or (3,3)
    
    An efficient closed-form solution of the matrix exponential for arguments
    that are so(2) or se(2).
    
    For so(2) the results is an SO(2) rotation matrix:

    - ``trexp2(S)`` is the matrix exponential of the so(3) element ``S`` which is a 2x2
      skew-symmetric matrix.
    - ``trexp2(S, THETA)`` as above but for an so(3) motion of S*THETA, where ``S`` is
      unit-norm skew-symmetric matrix representing a rotation axis and a rotation magnitude
      given by ``THETA``.
    - ``trexp2(W)`` is the matrix exponential of the so(2) element ``W`` expressed as
      a 1-vector (array_like).
    - ``trexp2(W, THETA)`` as above but for an so(3) motion of W*THETA where ``W`` is a
      unit-norm vector representing a rotation axis and a rotation magnitude
      given by ``THETA``. ``W`` is expressed as a 1-vector (array_like).


    For se(2) the results is an SE(2) homogeneous transformation matrix:

    - ``trexp2(SIGMA)`` is the matrix exponential of the se(2) element ``SIGMA`` which is
      a 3x3 augmented skew-symmetric matrix.
    - ``trexp2(SIGMA, THETA)`` as above but for an se(3) motion of SIGMA*THETA, where ``SIGMA``
      must represent a unit-twist, ie. the rotational component is a unit-norm skew-symmetric
      matrix.
    - ``trexp2(TW)`` is the matrix exponential of the se(3) element ``TW`` represented as
      a 3-vector which can be considered a screw motion.
    - ``trexp2(TW, THETA)`` as above but for an se(2) motion of TW*THETA, where ``TW``
      must represent a unit-twist, ie. the rotational component is a unit-norm skew-symmetric
      matrix.
          
     :seealso: trlog, trexp2
    """
    
    if argcheck.ismatrix(S, (3,3)) or argcheck.isvector(S, 3):
        # se(2) case
        if argcheck.ismatrix(S, (3,3)):
            # augmentented skew matrix
            tw = vexa(S)
        else:
            # 3 vector
            tw = argcheck.getvector(S)

        if theta is not None:
                assert isunittwist(tw), 'If theta is specified S must be a unit twist'
                
        t = tw[0:2]
        w = tw[2]
        
    elif argcheck.ismatrix(S, (2,2)) or argcheck.isvector(S, 1):
        # so(2) case
        if argcheck.ismatrix(S, (2,2)):
            # skew symmetric matrix
            w = vex(S)
        else:
            # 1 vector
            w = argcheck.getvector(S)
            
        if theta is not None:
            assert isunitvec(w), 'If theta is specified S must be a unit twist'
        t = None
    else:
        raise ValueError(" First argument must be SO(2), 1-vector, SE(2) or 3-vector")
    
    
    # do Rodrigues' formula for rotation
    if iszerovec(w):
        # for a zero so(2) return unit matrix, theta not relevant
        R = np.eye(2)
        V = np.eye(2)
    else:
        if theta is None:
            #  theta is not given, extract it
            theta = norm(w)
            w = unitvec(w)

        skw = skew(w)
        R = np.eye(2) + math.sin(theta) * skw + (1.0 - math.cos(theta)) * skw @ skw
        V = None
    
    if t is None:
        # so(2) case
        return R
    else:
        # se(3) case
        if V is None:
            V = np.eye(3) + (1.0-math.cos(theta))*skw/theta + (theta-math.sin(theta))/theta*skw @ skw
        return rt2tr(R, V@t)

    
def trprint2(T, label=None, file=sys.stdout, fmt='{:8.2g}', unit='deg'):
    """
    Compact display of SO(2) or SE(2) matrices
    
    :param T: matrix to format
    :type T: numpy.ndarray, shape=(2,2) or (3,3)
    :param label: text label to put at start of line
    :type label: str
    :param file: file to write formatted string to
    :type file: str
    :param fmt: conversion format for each number
    :type fmt: str
    :param unit: angular units: 'rad' [default], or 'deg'
    :type unit: str
    :return: optional formatted string
    :rtype: str
    
    The matrix is formatted and written to ``file`` or if ``file=None`` then the
    string is returned.
    
    - ``trprint2(R)`` displays the SO(2) rotation matrix in a compact 
      single-line format:
        
        [LABEL:] THETA UNIT
        
    - ``trprint2(T)`` displays the SE(2) homogoneous transform in a compact 
      single-line format:
        
        [LABEL:] [t=X, Y;] THETA UNIT

    Example:
        
    >>> T = transl2(1,2)@trot2(0.3)
    >>> trprint2(a, file=None, label='T')
    'T: t =        1,        2;       17 deg'

    :seealso: trprint
    """
    
    s = ''
    
    if label is not None:
        s += '{:s}: '.format(label)
    
    # print the translational part if it exists
    s += 't = {};'.format(_vec2s(fmt, transl2(T)))
    
    angle = math.atan2(T[1,0], T[0,0])
    if unit == 'deg':
        angle *= 180.0/math.pi
    s += ' {} {}'.format(_vec2s(fmt, [angle]), unit)
    
    if file:
        print(s, file=file)
    else:
        return s
    
def _vec2s(fmt, v):
        v = [x if np.abs(x) > 100*_eps else 0.0 for x in v ]
        return ', '.join([fmt.format(x) for x in v])
    
if __name__ == '__main__':
    import pathlib
    import os.path
    
    runfile(os.path.join(pathlib.Path(__file__).parent.absolute(), "test_transforms.py") )
    
    



