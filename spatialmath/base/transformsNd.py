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
# pylint: disable=invalid-name

import math
import numpy as np
from spatialmath.base import vectors as vec
from spatialmath.base import transforms2d as t2d
from spatialmath.base import transforms3d as t3d
from spatialmath.base import argcheck
from spatialmath.base import symbolic as sym

_eps = np.finfo(np.float64).eps

# ---------------------------------------------------------------------------------------#
def r2t(R, check=False):
    """
    Convert SO(n) to SE(n)

    :param R: rotation matrix
    :param check: check if rotation matrix is valid (default False, no check)
    :return: homogeneous transformation matrix
    :rtype: numpy.ndarray, shape=(3,3) or (4,4)

    ``T = r2t(R)`` is an SE(2) or SE(3) homogeneous transform equivalent to an
    SO(2) or SO(3) orthonormal rotation matrix ``R`` with a zero translational
    component

    - if ``R`` is 2x2 then ``T`` is 3x3: SO(2) -> SE(2)
    - if ``R`` is 3x3 then ``T`` is 4x4: SO(3) -> SE(3)

    :seealso: t2r, rt2tr
    """
    dim = R.shape
    assert dim[0] == dim[1], 'Matrix must be square'
    n = dim[0] + 1
    m = dim[0]

    if R.dtype == 'O':
        # symbolic matrix
        T = np.zeros((n, n), dtype='O')
    else:
        # numeric matrix
        assert isinstance(R, np.ndarray)
        if check and np.abs(np.linalg.det(R) - 1) < 100 * _eps:
            raise ValueError('Invalid rotation matrix ')

        # T = np.pad(R, (0, 1), mode='constant')
        # T[-1, -1] = 1.0
        T = np.zeros((n, n))
    T[:m,:m] = R
    T[-1, -1] = 1

    return T


# ---------------------------------------------------------------------------------------#
def t2r(T, check=False):
    """
    Convert SE(n) to SO(n)

    :param T: homogeneous transformation matrix
    :param check: check if rotation matrix is valid (default False, no check)
    :return: rotation matrix
    :rtype: numpy.ndarray, shape=(2,2) or (3,3)


    ``R = T2R(T)`` is the orthonormal rotation matrix component of homogeneous
    transformation matrix ``T``

    - if ``T`` is 3x3 then ``R`` is 2x2: SE(2) -> SO(2)
    - if ``T`` is 4x4 then ``R`` is 3x3: SE(3) -> SO(3)

    Any translational component of T is lost.

    :seealso: r2t, tr2rt
    """
    assert isinstance(T, np.ndarray)
    dim = T.shape
    assert dim[0] == dim[1], 'Matrix must be square'

    if dim[0] == 3:
        R = T[:2, :2]
    elif dim[0] == 4:
        R = T[:3, :3]
    else:
        raise ValueError('Value must be a rotation matrix')

    if check and isR(R):
        raise ValueError('Invalid rotation matrix')

    return R

# ---------------------------------------------------------------------------------------#


def tr2rt(T, check=False):
    """
    Convert SE(3) to SO(3) and translation

    :param T: homogeneous transform matrix
    :param check: check if rotation matrix is valid (default False, no check)
    :return: Rotation matrix and translation vector
    :rtype: tuple: numpy.ndarray, shape=(2,2) or (3,3); numpy.ndarray, shape=(2,) or (3,)

    (R,t) = tr2rt(T) splits a homogeneous transformation matrix (NxN) into an orthonormal
    rotation matrix R (MxM) and a translation vector T (Mx1), where N=M+1.

    - if ``T`` is 3x3 - in SE(2) - then ``R`` is 2x2 and ``t`` is 2x1.
    - if ``T`` is 4x4 - in SE(3) - then ``R`` is 3x3 and ``t`` is 3x1.

    :seealso: rt2tr, tr2r
    """
    dim = T.shape
    assert dim[0] == dim[1], 'Matrix must be square'

    if dim[0] == 3:
        R = t2r(T, check)
        t = T[:2, 2]
    elif dim[0] == 4:
        R = t2r(T, check)
        t = T[:3, 3]
    else:
        raise ValueError('T must be an SE2 or SE3 homogeneous transformation matrix')

    return [R, t]

# ---------------------------------------------------------------------------------------#


def rt2tr(R, t, check=False):
    """
    Convert SO(3) and translation to SE(3)

    :param R: rotation matrix
    :param t: translation vector
    :param check: check if rotation matrix is valid (default False, no check)
    :return: homogeneous transform
    :rtype: numpy.ndarray, shape=(3,3) or (4,4)

    ``T = rt2tr(R, t)`` is a homogeneous transformation matrix (N+1xN+1) formed from an
    orthonormal rotation matrix ``R`` (NxN) and a translation vector ``t``
    (Nx1).

    - If ``R`` is 2x2 and ``t`` is 2x1, then ``T`` is 3x3
    - If ``R`` is 3x3 and ``t`` is 3x1, then ``T`` is 4x4

    :seealso: rt2m, tr2rt, r2t
    """
    t = argcheck.getvector(t, dim=None, out='array')
    if R.shape[0] != t.shape[0]:
        raise ValueError("R and t must have the same number of rows")
    if check and np.abs(np.linalg.det(R) - 1) < 100 * _eps:
        raise ValueError('Invalid rotation matrix')

    if R.shape == (2, 2):
        T = np.eye(3)
        T[:2, :2] = R
        T[:2, 2] = t
    elif R.shape == (3, 3):
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
    else:
        raise ValueError('R must be an SO2 or SO3 rotation matrix')

    return T

# ---------------------------------------------------------------------------------------#


def rt2m(R, t, check=False):
    """
    Pack rotation and translation to matrix

    :param R: rotation matrix
    :param t: translation vector
    :param check: check if rotation matrix is valid (default False, no check)
    :return: homogeneous transform
    :rtype: numpy.ndarray, shape=(3,3) or (4,4)

    ``T = rt2m(R, t)`` is a matrix (N+1xN+1) formed from a matrix ``R`` (NxN) and a vector ``t``
    (Nx1).  The bottom row is all zeros.

    - If ``R`` is 2x2 and ``t`` is 2x1, then ``T`` is 3x3
    - If ``R`` is 3x3 and ``t`` is 3x1, then ``T`` is 4x4

    :seealso: rt2tr, tr2rt, r2t
    """
    t = argcheck.getvector(t, dim=None, out='array')
    if R.shape[0] != t.shape[0]:
        raise ValueError("R and t must have the same number of rows")
    if check and np.abs(np.linalg.det(R) - 1) < 100 * _eps:
        raise ValueError('Invalid rotation matrix')

    if R.shape == (2, 2):
        T = np.zeros((3, 3))
        T[:2, :2] = R
        T[:2, 2] = t
    elif R.shape == (3, 3):
        T = np.zeros((4, 4))
        T[:3, :3] = R
        T[:3, 3] = t
    else:
        raise ValueError('R must be an SO2 or SO3 rotation matrix')

    return T

# ======================= predicates


def isR(R, tol=100):
    r"""
    Test if matrix belongs to SO(n)

    :param R: matrix to test
    :type R: numpy.ndarray
    :param tol: tolerance in units of eps
    :type tol: float
    :return: whether matrix is a proper orthonormal rotation matrix
    :rtype: bool

    Checks orthogonality, ie. :math:`{\bf R} {\bf R}^T = {\bf I}` and :math:`\det({\bf R}) > 0`.
    For the first test we check that the norm of the residual is less than ``tol * eps``.

    :seealso: isrot2, isrot
    """
    return np.linalg.norm(R@R.T - np.eye(R.shape[0])) < tol * _eps \
        and np.linalg.det(R@R.T) > 0


def isskew(S, tol=10):
    r"""
    Test if matrix belongs to so(n)

    :param S: matrix to test
    :type S: numpy.ndarray
    :param tol: tolerance in units of eps
    :type tol: float
    :return: whether matrix is a proper skew-symmetric matrix
    :rtype: bool

    Checks skew-symmetry, ie. :math:`{\bf S} + {\bf S}^T = {\bf 0}`.
    We check that the norm of the residual is less than ``tol * eps``.

    :seealso: isskewa
    """
    return np.linalg.norm(S + S.T) < tol * _eps


def isskewa(S, tol=10):
    r"""
    Test if matrix belongs to se(n)

    :param S: matrix to test
    :type S: numpy.ndarray
    :param tol: tolerance in units of eps
    :type tol: float
    :return: whether matrix is a proper skew-symmetric matrix
    :rtype: bool

    Check if matrix is augmented skew-symmetric, ie. the top left (n-1xn-1) partition ``S`` is
    skew-symmetric :math:`{\bf S} + {\bf S}^T = {\bf 0}`, and the bottom row is zero
    We check that the norm of the residual is less than ``tol * eps``.

    :seealso: isskew
    """
    return np.linalg.norm(S[0:-1, 0:-1] + S[0:-1, 0:-1].T) < tol * _eps \
        and np.all(S[-1, :] == 0)


def iseye(S, tol=10):
    """
    Test if matrix is identity

    :param S: matrix to test
    :type S: numpy.ndarray
    :param tol: tolerance in units of eps
    :type tol: float
    :return: whether matrix is a proper skew-symmetric matrix
    :rtype: bool

    Check if matrix is an identity matrix. We test that the trace tom row is zero
    We check that the norm of the residual is less than ``tol * eps``.

    :seealso: isskew, isskewa
    """
    s = S.shape
    if len(s) != 2 or s[0] != s[1]:
        return False  # not a square matrix
    return vec.norm(S - np.eye(s[0])) < tol * _eps


# ========================= angle sequences


# ---------------------------------------------------------------------------------------#
def skew(v):
    r"""
    Create skew-symmetric metrix from vector

    :param v: 1- or 3-vector
    :type v: array_like
    :return: skew-symmetric matrix in so(2) or so(3)
    :rtype: numpy.ndarray, shape=(2,2) or (3,3)
    :raises: ValueError

    ``skew(V)`` is a skew-symmetric matrix formed from the elements of ``V``.

    - ``len(V)``  is 1 then ``S`` = :math:`\left[ \begin{array}{cc} 0 & -v \\ v & 0 \end{array} \right]`
    - ``len(V)`` is 3 then ``S`` = :math:`\left[ \begin{array}{ccc} 0 & -v_z & v_y \\ v_z & 0 & -v_x \\ -v_y & v_x & 0\end{array} \right]`

    Notes:

    - This is the inverse of the function ``vex()``.
    - These are the generator matrices for the Lie algebras so(2) and so(3).

    :seealso: vex, skewa
    """
    v = argcheck.getvector(v, None, 'sequence')
    if len(v) == 1:
        s = np.array([
            [0, -v[0]],
            [v[0], 0]])
    elif len(v) == 3:
        s = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]])
    else:
        raise AttributeError("argument must be a 1- or 3-vector")

    return s

# ---------------------------------------------------------------------------------------#


def vex(s):
    r"""
    Convert skew-symmetric matrix to vector

    :param s: skew-symmetric matrix
    :type s: numpy.ndarray, shape=(2,2) or (3,3)
    :return: vector of unique values
    :rtype: numpy.ndarray, shape=(1,) or (3,)
    :raises: ValueError

    ``vex(S)`` is the vector which has the corresponding skew-symmetric matrix ``S``.

    - ``S`` is 2x2 - so(2) case - where ``S`` :math:`= \left[ \begin{array}{cc} 0 & -v \\ v & 0 \end{array} \right]` then return :math:`[v]`
    - ``S`` is 3x3 - so(3) case -  where ``S`` :math:`= \left[ \begin{array}{ccc} 0 & -v_z & v_y \\ v_z & 0 & -v_x \\ -v_y & v_x & 0\end{array} \right]` then return :math:`[v_x, v_y, v_z]`.

    Notes:

    - This is the inverse of the function ``skew()``.
    - Only rudimentary checking (zero diagonal) is done to ensure that the matrix
      is actually skew-symmetric.
    - The function takes the mean of the two elements that correspond to each unique
      element of the matrix.

    :seealso: skew, vexa
    """
    if s.shape == (3, 3):
        return 0.5 * np.array([s[2, 1] - s[1, 2], s[0, 2] - s[2, 0], s[1, 0] - s[0, 1]])
    elif s.shape == (2, 2):
        return 0.5 * np.array([s[1, 0] - s[0, 1]])
    else:
        raise ValueError("Argument must be 2x2 or 3x3 matrix")

# ---------------------------------------------------------------------------------------#


def skewa(v):
    r"""
    Create augmented skew-symmetric metrix from vector

    :param v: 3- or 6-vector
    :type v: array_like
    :return: augmented skew-symmetric matrix in se(2) or se(3)
    :rtype: numpy.ndarray, shape=(3,3) or (4,4)
    :raises: ValueError

    ``skewa(V)`` is an augmented skew-symmetric matrix formed from the elements of ``V``.

    - ``len(V)`` is 3 then S = :math:`\left[ \begin{array}{ccc} 0 & -v_3 & v_1 \\ v_3 & 0 & v_2 \\ 0 & 0 & 0 \end{array} \right]`
    - ``len(V)`` is 6 then S = :math:`\left[ \begin{array}{cccc} 0 & -v_6 & v_5 & v_1 \\ v_6 & 0 & -v_4 & v_2 \\ -v_5 & v_4 & 0 & v_3 \\ 0 & 0 & 0 & 0 \end{array} \right]`

    Notes:

    - This is the inverse of the function ``vexa()``.
    - These are the generator matrices for the Lie algebras se(2) and se(3).
    - Map twist vectors in 2D and 3D space to se(2) and se(3).

    :seealso: vexa, skew
    """

    v = argcheck.getvector(v, None, 'sequence')
    if len(v) == 3:
        omega = np.zeros((3, 3))
        omega[:2, :2] = skew(v[2])
        omega[:2, 2] = v[0:2]
        return omega
    elif len(v) == 6:
        omega = np.zeros((4, 4))
        omega[:3, :3] = skew(v[3:6])
        omega[:3, 3] = v[0:3]
        return omega
    else:
        raise AttributeError("expecting a 3- or 6-vector")


def vexa(Omega):
    r"""
    Convert skew-symmetric matrix to vector

    :param s: augmented skew-symmetric matrix
    :type s: numpy.ndarray, shape=(3,3) or (4,4)
    :return: vector of unique values
    :rtype: numpy.ndarray, shape=(3,) or (6,)
    :raises: ValueError

    ``vex(S)`` is the vector which has the corresponding skew-symmetric matrix ``S``.

    - ``S`` is 3x3 - se(2) case - where ``S`` :math:`= \left[ \begin{array}{ccc} 0 & -v_3 & v_1 \\ v_3 & 0 & v_2 \\ 0 & 0 & 0 \end{array} \right]` then return :math:`[v_1, v_2, v_3]`.
    - ``S`` is 4x4 - se(3) case -  where ``S`` :math:`= \left[ \begin{array}{cccc} 0 & -v_6 & v_5 & v_1 \\ v_6 & 0 & -v_4 & v_2 \\ -v_5 & v_4 & 0 & v_3 \\ 0 & 0 & 0 & 0 \end{array} \right]` then return :math:`[v_1, v_2, v_3, v_4, v_5, v_6]`.


    Notes:

    - This is the inverse of the function ``skewa``.
    - Only rudimentary checking (zero diagonal) is done to ensure that the matrix
      is actually skew-symmetric.
    - The function takes the mean of the two elements that correspond to each unique
      element of the matrix.

    :seealso: skewa, vex
    """
    if Omega.shape == (4, 4):
        return np.hstack((t3d.transl(Omega), vex(t2r(Omega))))
    elif Omega.shape == (3, 3):
        return np.hstack((t2d.transl2(Omega), vex(t2r(Omega))))
    else:
        raise AttributeError("expecting a 3x3 or 4x4 matrix")


def rodrigues(w, theta):
    """
    Rodrigues' formula for rotation

    :param w: rotation vector
    :type w: array_like
    :param theta: rotation angle
    :type theta: float or None
    """
    w = argcheck.getvector(w)
    if vec.iszerovec(w):
        # for a zero so(n) return unit matrix, theta not relevant
        if len(w) == 1:
            return np.eye(2)
        else:
            return np.eye(3)
    if theta is None:
        theta = vec.norm(w)
        w = vec.unitvec(w)

    skw = skew(w)
    return np.eye(skw.shape[0]) + math.sin(theta) * skw + (1.0 - math.cos(theta)) * skw @ skw


def h2e(v):
    """
    Convert from homogeneous to Euclidean form

    :param v: homogeneous vector or matrix
    :type v: array_like
    :return: Euclidean vector
    :rtype: numpy.ndarray

    - If ``v`` is an array, shape=(N,), return an array shape=(N-1,) where the elements have
      all been scaled by the last element of ``v``.
    - If ``v`` is a matrix, shape=(N,M), return a matrix shape=(N-1,N), where each column has
      been scaled by its last element.

    :seealso: e2h
    """
    if argcheck.isvector(v):
        # dealing with shape (N,) array
        v = argcheck.getvector(v)
        return v[0:-1] / v[-1]
    elif isinstance(v, np.ndarray) and len(v.shape) == 2:
        # dealing with matrix
        return v[:-1, :] / np.tile(v[-1, :], (v.shape[0] - 1, 1))


def e2h(v):
    """
    Convert from Euclidean to homogeneous form

    :param v: Euclidean vector or matrix
    :type v: array_like
    :return: homogeneous vector
    :rtype: numpy.ndarray

    - If ``v`` is an array, shape=(N,), return an array shape=(N+1,) where a value of 1 has
      been appended
    - If ``v`` is a matrix, shape=(N,M), return a matrix shape=(N+1,N), where each column has
      been appended with a value of 1, ie. a row of ones has been appended to the matrix.

    :seealso: e2h
    """
    if argcheck.isvector(v):
        # dealing with shape (N,) array
        v = argcheck.getvector(v)
        return np.r_[v, 1]
    elif isinstance(v, np.ndarray) and len(v.shape) == 2:
        # dealing with matrix
        return np.vstack([v, np.ones((1, v.shape[1]))])

def homtrans(T, p):
    r"""
    Apply a homogeneous transformation to a Euclidean vector

    :param T: homogeneous transformation
    :type T: Numpy array (3,3) or (4,4)
    :param p: Vector(s) to be transformed
    :type p: Numpy array (2,), (2,N), (3,) or (3,N)
    :return: transformed Euclidean vector(s)
    :rtype: Numpy array (2,), (2,N), (3,) or (3,N)

    ``homtrans(T, p)`` applies the homogeneous transformation ``T`` to the points 
    stored columnwise in ``p``.

    - If ``T`` is in SE(2) (3x3) and
        - ``p`` is 2xN (2D points) they are considered Euclidean (:math:`\mathbb{R}^2`)
        - ``p`` is 3xN (2D points) they are considered projective (:math:`\mathbb{P}^2`)
    - If ``T`` is in SE(3) (4x4) and
        - ``p`` is 3xN (3D points) they are considered Euclidean (:math:`\mathbb{R}^3`)
        - ``p`` is 4xN (3D points) they are considered projective (:math:`\mathbb{P}^3`)

    The return value and ``p`` have the same number of rows, ie. if Euclidean points are given
    then Euclidean points are returned, if projective points are given then
    projective points are returned.

    Notes:
    - If T is a homogeneous transformation defining the pose of {B} with respect to {A},
    then the points are defined with respect to frame {B} and are transformed to be
    with respect to frame {A}.

    :seealso: :func:`e2h`, :func:`h2e`
    """
    if p.shape[0] == T.shape[0] - 1:
        # Euclidean vector
        return h2e( T @ e2h(p) )
    elif p.shape[0] == T.shape[0]:
        # homogeneous vector
        return T @ p
    else:
        raise ValueError('matrices and point data do not conform')

if __name__ == '__main__':  # pragma: no cover
    import pathlib

    exec(open(pathlib.Path(__file__).parent.absolute() / "test_transforms.py").read())  # pylint: disable=exec-used
