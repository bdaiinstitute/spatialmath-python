# Part of Spatial Math Toolbox for Python
# Copyright (c) 2000 Peter Corke
# MIT Licence, see details in top-level file: LICENCE

# pylint: disable=invalid-name

import sys
import math
import numpy as np
from spatialmath import base

_eps = np.finfo(np.float64).eps


def eye():
    """
    Create an identity quaternion

    :return: an identity quaternion
    :rtype: ndarray(4)

    Creates an identity quaternion, with the scalar part equal to one, and
    a zero vector value.

    .. runblock:: pycon

        >>> from spatialmath.base import eye, qprint
        >>> q = eye()
        >>> qprint(q)

    """
    return np.r_[1, 0, 0, 0]


def pure(v):
    """
    Create a pure quaternion

    :arg v: 3D vector
    :type v: array_like(3)
    :return: pure quaternion
    :rtype: ndarray(4)

    Creates a pure quaternion, with a zero scalar value and the vector part
    equal to the passed vector value.

    .. runblock:: pycon

        >>> from spatialmath.base import pure, qprint
        >>> q = pure([1, 2, 3])
        >>> qprint(q)
    """
    v = base.getvector(v, 3)
    return np.r_[0, v]


def qpositive(q):
    """
    Quaternion with positive scalar part

    :arg q: quaternion
    :type v: : ndarray(4)
    :return: pure quaternion
    :rtype: ndarray(4)

    If the scalar part is negative return -q.
    """
    if q[0] < 0:
        return -q
    else:
        return q


def qnorm(q):
    r"""
    Norm of a quaternion

    :arg q: quaternion
    :type v: : array_like(4)
    :return: norm of the quaternion
    :rtype: float

    Returns the norm, length or magnitude of the input quaternion which is
    :math:`(s^2 + v_x^2 + v_y^2 + v_z^2}^{1/2}`

    .. runblock:: pycon

        >>> from spatialmath.base import qnorm
        >>> q = qnorm([1, 2, 3, 4])
        >>> print(q)

    :seealso: unit

    """
    q = base.getvector(q, 4)
    return np.linalg.norm(q)


def unit(q, tol=10):
    """
    Create a unit quaternion

    :arg v: quaterion
    :type v: array_like(4)
    :return: a pure quaternion
    :rtype: ndarray(4)
    :raises ValueError: quaternion has (near) zero norm

    Creates a unit quaternion, with unit norm, by scaling the input quaternion.

    .. runblock:: pycon

        >>> from spatialmath.base import unit, qprint
        >>> q = unit([1, 2, 3, 4])
        >>> qprint(q)

    .. note:: Scalar part is always positive.

    .. note:: If the quaternion norm is less than ``tol * eps`` an exception is
              raised.

    :seealso: norm
    """
    q = base.getvector(q, 4)
    nm = np.linalg.norm(q)
    if abs(nm) < tol * _eps:
        raise ValueError("cannot normalize (near) zero length quaternion")
    else:
        q /= nm

    if q[0] >= 0:
        return q
    else:
        return -q
    # return q


def isunit(q, tol=100):
    """
    Test if quaternion has unit length

    :param v: quaternion
    :type v: array_like(4)
    :param tol: tolerance in units of eps
    :type tol: float
    :return: whether quaternion has unit length
    :rtype: bool

    .. runblock:: pycon

        >>> from spatialmath.base import eye, pure, isunit
        >>> q = eye()
        >>> isunit(q)
        >>> q = pure([1, 2, 3])
        >>> isunit(q)

    :seealso: unit
    """
    return base.iszerovec(q, tol=tol)


def isequal(q1, q2, tol=100, unitq=False):
    """
    Test if quaternions are equal

    :param q1: quaternion
    :type q1: array_like(4)
    :param q2: quaternion
    :type q2: array_like(4)
    :param unitq: quaternions are unit quaternions
    :type unitq: bool
    :param tol: tolerance in units of eps
    :type tol: float
    :return: whether quaternions are equal
    :rtype: bool

    Tests if two quaternions are equal.

    For unit-quaternions ``unitq=True`` the double mapping is taken into account,
    that is ``q`` and ``-q`` represent the same orientation and ``isequal(q, -q, unitq=True)`` will
    return ``True``.

    .. runblock:: pycon

        >>> from spatialmath.base import isequal
        >>> q1 = [1, 2, 3, 4]
        >>> q2 = [-1, -2, -3, -4]
        >>> isequal(q1, q2)
        >>> isequal(q1, q2, unitq=True)
    """
    q1 = base.getvector(q1, 4)
    q2 = base.getvector(q2, 4)

    if unitq:
        return (np.sum(np.abs(q1 - q2)) < tol * _eps) or (np.sum(np.abs(q1 + q2)) < tol * _eps)
    else:
        return np.sum(np.abs(q1 - q2)) < tol * _eps


def q2v(q):
    """
    Convert unit-quaternion to 3-vector

    :arg q: unit-quaternion
    :type v: array_like(4)
    :return: a unique 3-vector
    :rtype: ndarray(3)

    Returns a unique 3-vector representing the input unit-quaternion. The sign
    of the scalar part is made positive, if necessary by multiplying the
    entire quaternion by -1, then the vector part is taken.

    .. runblock:: pycon

        >>> from spatialmath.base import q2v
        >>> from math import sqrt
        >>> q = [1 / sqrt(2), 0, 1 / sqrt(2), 0]
        >>> print(q2v(q))
        >>> q = [-1 / sqrt(2), 0, 1 / sqrt(2), 0]
        >>> print(q2v(q))

    .. warning:: There is no check that the passed value is a unit-quaternion.

    :seealso: :func:`~v2q`

    """
    q = base.getvector(q, 4)
    if q[0] >= 0:
        return q[1:4]
    else:
        return -q[1:4]


def v2q(v):
    r"""
    Convert 3-vector to unit-quaternion

    :arg v: vector part of unit quaternion
    :type v: array_like(3)
    :return: a unit quaternion
    :rtype: ndarray(4)

    Returns a unit-quaternion reconsituted from just its vector part.  Assumes
    that the scalar part was positive, so :math:`s = \sqrt{1-||v||}`.

    .. runblock:: pycon

        >>> from spatialmath.base import v2q, qprint
        >>> from math import sqrt
        >>> v = [0, 1 / sqrt(2), 0]
        >>> qprint(v2q(v))
        >>> v = [0, -1 / sqrt(2), 0]
        >>> qprint(v2q(v))

    .. warning:: There is no check that the value is the vector part of
                 a unit-quaternion, and this can lead to a math domain error.

    :seealso: :func:`q2v`
    """
    v = base.getvector(v, 3)
    s = math.sqrt(1 - np.sum(v**2))
    return np.r_[s, v]


def qqmul(q1, q2):
    """
    Quaternion multiplication

    :arg q0: left-hand quaternion
    :type q0: : array_like(4)
    :arg q1: right-hand quaternion
    :type q1: array_like(4)
    :return: quaternion product
    :rtype: ndarray(4)

    This is the quaternion or Hamilton product.  If both operands are unit-quaternions then
    the product will be a unit-quaternion.

    .. runblock:: pycon

        >>> from spatialmath.base import qqmul
        >>> q1 = [1, 2, 3, 4]
        >>> q2 = [5, 6, 7, 8]
        >>> qqmul(q1, q2)    # conventional Hamilton product

    :seealso: qvmul, inner, vvmul

    """
    q1 = base.getvector(q1, 4)
    q2 = base.getvector(q2, 4)
    s1 = q1[0]
    v1 = q1[1:4]
    s2 = q2[0]
    v2 = q2[1:4]

    return np.r_[s1 * s2 - np.dot(v1, v2), s1 * v2 + s2 * v1 + np.cross(v1, v2)]


def inner(q1, q2):
    """
    Quaternion inner product

    :arg q0: quaternion 
    :type q0: : array_like(4)
    :arg q1: uaternion
    :type q1: array_like(4)
    :return: inner product
    :rtype: ndarray(4)

    This is the inner or dot product of two quaternions, it is the sum of the element-wise
    product.

    - The inner product ``inner(q, q)`` is the square of the norm of ``q``.
    - If ``q0`` and ``q1`` are unit quaternions then the inner product is the
      cosine of the angle between the two orientations.

    .. runblock:: pycon

        >>> from spatialmath.base import inner
        >>> from math import sqrt, acos, pi
        >>> q1 = [1, 2, 3, 4]
        >>> inner(q1, q1)                      # square of the norm
        >>> q1 = [1/sqrt(2), 1/sqrt(2), 0, 0]  # 90deg rotation about x-axis
        >>> q2 = [1/sqrt(2), 0, 1/sqrt(2), 0]  # 90deg rotation about y-axis
        >>> acos(inner(q1, q2)) * 180 / pi     # angle between q1 and q2

    :seealso: qvmul

    """
    q1 = base.getvector(q1, 4)
    q2 = base.getvector(q2, 4)

    return np.dot(q1, q2)


def qvmul(q, v):
    """
    Vector rotation

    :arg q: unit-quaternion
    :type q: array_like(4)
    :arg v: 3-vector to be rotated
    :type v: array_like(3)
    :return: rotated 3-vector
    :rtype: ndarray(3)

    The vector `v` is rotated about the origin by the SO(3) equivalent of the unit
    quaternion.

    .. runblock:: pycon

        >>> from spatialmath.base import qvmul
        >>> from math import sqrt
        >>> q = [1/sqrt(2), 1/sqrt(2), 0, 0]  # 90deg rotation about x-axis
        >>> qvmul(q, [1, 2, 3])              # rotated vector

    .. warning:: There is no check that the passed value is a unit-quaternions.

    :seealso: qvmul
    """
    q = base.getvector(q, 4)
    v = base.getvector(v, 3)
    qv = qqmul(q, qqmul(pure(v), conj(q)))
    return qv[1:4]


def vvmul(qa, qb):
    """
    Quaternion multiplication


    :arg qa: left-hand quaternion
    :type qa: : array_like(3)
    :arg qb: right-hand quaternion
    :type qb: array_like(3)
    :return: quaternion product
    :rtype: ndarray(3)

    This is the quaternion or Hamilton product of unit-quaternions defined only
    by their vector components.  The product will be a unit-quaternion, defined only
    by its vector component.

    .. runblock:: pycon

        >>> from spatialmath.base import vvmul, v2q, q2v, qqmul, qprint
        >>> from math import sqrt
        >>> q1 = [1/sqrt(2), 1/sqrt(2), 0, 0]  # 90deg rotation about x-axis
        >>> q2 = [1/sqrt(2), 0, 1/sqrt(2), 0]  # 90deg rotation about y-axis
        >>> qprint(qqmul(q1, q2))              # normal Hamilton product
        >>> v1 = q2v(q1); v2 = q2v(q2)
        >>> vp = vvmul(v1, v2)                 # product using 3-vectors
        >>> qprint(v2q(vp))                    # same answer as Hamilton product

    :seealso: :func:`q2v`, :func:`v2q`, :func:`qvmul`
    """
    t6 = math.sqrt(1.0 - np.sum(qa**2))
    t11 = math.sqrt(1.0 - np.sum(qb**2))
    return np.r_[qa[1] * qb[2] - qb[1] * qa[2] + qb[0] * t6 + qa[0] * t11, -qa[0] * qb[2] + qb[0] * qa[2] + qb[1] * t6 + qa[1] * t11, qa[0] * qb[1] - qb[0] * qa[1] + qb[2] * t6 + qa[2] * t11]


def qpow(q, power):
    """
    Raise quaternion to a power

    :arg q: quaternion
    :type v: array_like(4)
    :arg power: exponent
    :type power: int
    :return: input quaternion raised to the specified power
    :rtype: ndarray(4)
    :raises ValueError: if exponent is non integer

    Raises a quaternion to the specified power using repeated multiplication.

    .. runblock:: pycon

        >>> from spatialmath.base import qpow, qqmul, qprint
        >>> q = [1, 2, 3, 4]
        >>> qprint(qqmul(q, q))
        >>> qprint(qpow(q, 2))
        >>> qprint(qpow(q, -2)) # conjugate of above

    .. note:

        - Power must be an integer
        - Power can be negative, in which case the conjugate is taken

    :seealso: :func:`qqmul`
    :SymPy: supported for ``q`` but not ``power``.
    """
    q = base.getvector(q, 4)
    if not isinstance(power, int):
        raise ValueError("Power must be an integer")
    qr = eye()
    for _ in range(0, abs(power)):
        qr = qqmul(qr, q)

    if power < 0:
        qr = conj(qr)

    return qr


def conj(q):
    """
    Quaternion conjugate

    :arg q: quaternion
    :type v: array_like(4)
    :return: conjugate of input quaternion
    :rtype: ndarray(4)

    Conjugate of quaternion, the vector part is negated.

    .. runblock:: pycon

        >>> from spatialmath.base import conj, qprint
        >>> q = [1, 2, 3, 4]
        >>> qprint(conj(q))

    :SymPy: supported
    """
    q = base.getvector(q, 4)
    return np.r_[q[0], -q[1:4]]


def q2r(q):
    """
    Convert unit-quaternion to SO(3) rotation matrix

    :arg q: unit-quaternion
    :type v: array_like(4)
    :return: corresponding SO(3) rotation matrix
    :rtype: ndarray(3,3)

    Returns an SO(3) rotation matrix corresponding to this unit-quaternion.

    .. runblock:: pycon

        >>> from spatialmath.base import q2r
        >>> q = [0, 0, 1, 0]  # rotation of 180deg about y-axis
        >>> print(q2r(q))

    .. warning:: There is no check that the passed value is a unit-quaternion.

    :seealso: :func:`r2q`

    """
    q = base.getvector(q, 4)
    s = q[0]
    x = q[1]
    y = q[2]
    z = q[3]
    return np.array([[1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - s * z), 2 * (x * z + s * y)],
                     [2 * (x * y + s * z), 1 - 2 *
                      (x ** 2 + z ** 2), 2 * (y * z - s * x)],
                     [2 * (x * z - s * y), 2 * (y * z + s * x), 1 - 2 * (x ** 2 + y ** 2)]])


def r2q(R, check=False, tol=100, order='sxyz'):
    """
    Convert SO(3) rotation matrix to unit-quaternion

    :arg R: SO(3) rotation matrix
    :type R: ndarray(3,3)
    :param check: check validity of rotation matrix, default False
    :type check: bool
    :param tol: tolerance in units of eps
    :type tol: float
    :param order: the order of the returned quaternion. Must be 'sxyz' or
        'xyzs'. Defaults to 'sxyz'.
    :type order: str
    :return: unit-quaternion as Euler parameters
    :rtype: ndarray(4)
    :raises ValueError: for non SO(3) argument

    Returns a unit-quaternion corresponding to the input SO(3) rotation matrix.

    .. runblock:: pycon

        >>> from spatialmath.base import r2q, qprint, rotx
        >>> R = rotx(90, 'deg') # rotation of 90deg about x-axis
        >>> print(R)
        >>> qprint(r2q(R))

    .. warning:: There is no check that the passed matrix is a valid rotation matrix.

    .. note:: 
        - Scalar part is always positive
        - implements Cayley's method

    :reference: 
        - Sarabandi, S., and Thomas, F. (March 1, 2019). 
          "A Survey on the Computation of Quaternions From Rotation Matrices." 
          ASME. J. Mechanisms Robotics. April 2019; 11(2): 021006. 
          `doi.org/10.1115/1.4041889 <https://doi.org/10.1115/1.4041889>`_

    :seealso: :func:`q2r`
    """
    if not base.isrot(R, check=check, tol=tol):
        raise ValueError("Argument must be a valid SO(3) matrix")

    t12p = (R[0, 1] + R[1, 0]) ** 2
    t13p = (R[0, 2] + R[2, 0]) ** 2
    t23p = (R[1, 2] + R[2, 1]) ** 2

    t12m = (R[0, 1] - R[1, 0]) ** 2
    t13m = (R[0, 2] - R[2, 0]) ** 2
    t23m = (R[1, 2] - R[2, 1]) ** 2

    d1 = (R[0, 0] + R[1, 1] + R[2, 2] + 1) ** 2
    d2 = (R[0, 0] - R[1, 1] - R[2, 2] + 1) ** 2
    d3 = (-R[0, 0] + R[1, 1] - R[2, 2] + 1) ** 2
    d4 = (-R[0, 0] - R[1, 1] + R[2, 2] + 1) ** 2

    e0 = math.sqrt(d1 + t23m + t13m + t12m) / 4.0
    e1 = math.sqrt(t23m + d2 + t12p + t13p) / 4.0
    e2 = math.sqrt(t13m + t12p + d3 + t23p) / 4.0
    e3 = math.sqrt(t12m + t13p + t23p + d4) / 4.0

    # transfer sign from rotation element differences
    if R[2, 1] < R[1, 2]:
        e1 = -e1
    if R[0, 2] < R[2, 0]:
        e2 = -e2
    if R[1, 0] < R[0, 1]:
        e3 = -e3

    if order == 'sxyz':
        return np.r_[e0, e1, e2, e3]
    elif order == 'xyzs':
        return np.r_[e1, e2, e3, e0]
    else:
        raise ValueError("order is invalid, must be 'sxyz' or 'xyzs'")

# def r2q_old(R, check=False, tol=100):
#     """
#     Convert SO(3) rotation matrix to unit-quaternion

#     :arg R: SO(3) rotation matrix
#     :type R: ndarray(3,3)
#     :param check: check validity of rotation matrix, default False
#     :type check: bool
#     :param tol: tolerance in units of eps
#     :type tol: float
#     :return: unit-quaternion
#     :rtype: ndarray(4)
#     :raises ValueError: for non SO(3) argument

#     Returns a unit-quaternion corresponding to the input SO(3) rotation matrix.

#     .. runblock:: pycon

#         >>> from spatialmath.base import r2q, qprint, rotx
#         >>> R = rotx(90, 'deg') # rotation of 90deg about x-axis
#         >>> print(R)
#         >>> qprint(r2q(R))

#     .. warning:: There is no check that the passed matrix is a valid rotation matrix.

#     .. note:: Scalar part is always positive.

#     :seealso: :func:`q2r`
#     """
#     if not base.isrot(R, check=check, tol=tol):
#         raise ValueError("Argument must be a valid SO(3) matrix")

#     qs = math.sqrt(max(0, np.trace(R) + 1)) / 2.0  # scalar part
#     kx = R[2, 1] - R[1, 2]  # Oz - Ay
#     ky = R[0, 2] - R[2, 0]  # Ax - Nz
#     kz = R[1, 0] - R[0, 1]  # Ny - Ox

#     if (R[0, 0] >= R[1, 1]) and (R[0, 0] >= R[2, 2]):
#         kx1 = R[0, 0] - R[1, 1] - R[2, 2] + 1  # Nx - Oy - Az + 1
#         ky1 = R[1, 0] + R[0, 1]  # Ny + Ox
#         kz1 = R[2, 0] + R[0, 2]  # Nz + Ax
#         add = (kx >= 0)
#     elif R[1, 1] >= R[2, 2]:
#         kx1 = R[1, 0] + R[0, 1]  # Ny + Ox
#         ky1 = R[1, 1] - R[0, 0] - R[2, 2] + 1  # Oy - Nx - Az + 1
#         kz1 = R[2, 1] + R[1, 2]  # Oz + Ay
#         add = (ky >= 0)
#     else:
#         kx1 = R[2, 0] + R[0, 2]  # Nz + Ax
#         ky1 = R[2, 1] + R[1, 2]  # Oz + Ay
#         kz1 = R[2, 2] - R[0, 0] - R[1, 1] + 1  # Az - Nx - Oy + 1
#         add = (kz >= 0)

#     if add:
#         kx = kx + kx1
#         ky = ky + ky1
#         kz = kz + kz1
#     else:
#         kx = kx - kx1
#         ky = ky - ky1
#         kz = kz - kz1

#     kv = np.r_[kx, ky, kz]
#     nm = np.linalg.norm(kv)
#     if abs(nm) < tol * _eps:
#         return eye()
#     else:
#         return np.r_[qs, (math.sqrt(1.0 - qs ** 2) / nm) * kv]


def slerp(q0, q1, s, shortest=False):
    """
    Quaternion conjugate

    :arg q0: initial unit quaternion
    :type q0: array_like(4)
    :arg q1: final unit quaternion
    :type q1: array_like(4)
    :arg s: interpolation coefficient in the range [0,1]
    :type s: float
    :arg shortest: choose shortest distance [default False]
    :type shortest: bool
    :return: interpolated unit-quaternion
    :rtype: ndarray(4)
    :raises ValueError: s is outside interval [0, 1]

    An interpolated quaternion between ``q0`` when ``s`` = 0 to ``q1`` when ``s`` = 1.

    Interpolation is performed on a great circle on a 4D hypersphere. This is
    a rotation about a single fixed axis in space which yields the straightest
    and shortest path between two points.

    For large rotations the path may be the *long way around* the circle,
    the option ``'shortest'`` ensures always the shortest path.

    .. runblock:: pycon

        >>> from spatialmath.base import slerp, qprint
        >>> from math import sqrt
        >>> q0 = [1/sqrt(2), 1/sqrt(2), 0, 0]  # 90deg rotation about x-axis
        >>> q1 = [1/sqrt(2), 0, 1/sqrt(2), 0]  # 90deg rotation about y-axis
        >>> qprint(slerp(q0, q1, 0))           # this is q0
        >>> qprint(slerp(q0, q1, 1))           # this is q1
        >>> qprint(slerp(q0, q1, 0.5))         # this is in "half way" between

    .. warning:: There is no check that the passed values are unit-quaternions.

    """
    if not 0 <= s <= 1:
        raise ValueError("s must be in the interval [0,1]")
    q0 = base.getvector(q0, 4)
    q1 = base.getvector(q1, 4)

    if s == 0:
        return q0
    elif s == 1:
        return q1

    dotprod = np.dot(q0, q1)

    # If the dot product is negative, the quaternions
    # have opposite handed-ness and slerp won't take
    # the shorter path. Fix by reversing one quaternion.
    if shortest:
        if dotprod < 0:
            q0 = -q0   # pylint: disable=invalid-unary-operand-type
            dotprod = -dotprod  # pylint: disable=invalid-unary-operand-type

    dotprod = np.clip(dotprod, -1, 1)  # Clip within domain of acos()
    theta = math.acos(dotprod)  # theta is the angle between rotation vectors
    if abs(theta) > 10 * _eps:
        s0 = math.sin((1 - s) * theta)
        s1 = math.sin(s * theta)
        return ((q0 * s0) + (q1 * s1)) / math.sin(theta)
    else:
        # quaternions are identical
        return q0


def rand():
    """
    Random unit-quaternion

    :return: random unit-quaternion
    :rtype: ndarray(4)

    Computes a uniformly distributed random unit-quaternion which can be
    considered equivalent to a random SO(3) rotation.

    .. runblock:: pycon

        >>> from spatialmath.base import rand, qprint
        >>> qprint(rand())
    """
    u = np.random.uniform(
        low=0, high=1, size=3)  # get 3 random numbers in [0,1]
    return np.r_[
        math.sqrt(1 - u[0]) * math.sin(2 * math.pi * u[1]),
        math.sqrt(1 - u[0]) * math.cos(2 * math.pi * u[1]),
        math.sqrt(u[0]) * math.sin(2 * math.pi * u[2]),
        math.sqrt(u[0]) * math.cos(2 * math.pi * u[2])]


def matrix(q):
    """
    Convert to 4x4 matrix equivalent

    :arg q: quaternion
    :type v: array_like(4)
    :return: equivalent matrix
    :rtype: ndarray(4)

    Hamilton multiplication between two quaternions can be considered as a
    matrix-vector product, the left-hand quaternion is represented by an
    equivalent 4x4 matrix and the right-hand quaternion as 4x1 column vector.

    .. runblock:: pycon

        >>> from spatialmath.base import matrix, qqmul, qprint
        >>> q1 = [1, 2, 3, 4]
        >>> q2 = [5, 6, 7, 8]
        >>> qqmul(q1, q2)    # conventional Hamilton product
        >>> m = matrix(q1)
        >>> print(m)
        >>> v = m @ np.array(q2)
        >>> print(v)

    :seealso: qqmul

    """
    q = base.getvector(q, 4)
    s = q[0]
    x = q[1]
    y = q[2]
    z = q[3]
    return np.array([[s, -x, -y, -z],
                     [x, s, -z, y],
                     [y, z, s, -x],
                     [z, -y, x, s]])


def dot(q, w):
    """
    Rate of change of unit-quaternion

    :arg q0: unit-quaternion
    :type q0: array_like(4)
    :arg w: 3D angular velocity in world frame
    :type w: array_like(3)
    :return: rate of change of unit quaternion
    :rtype: ndarray(4)

    ``dot(q, w)`` is the rate of change of the elements of the unit quaternion ``q``
    which represents the orientation of a body frame with angular velocity ``w`` in
    the world frame.

    .. runblock:: pycon

        >>> from spatialmath.base import dot, qprint
        >>> from math import sqrt
        >>> q = [1/sqrt(2), 1/sqrt(2), 0, 0]   # 90deg rotation about x-axis
        >>> dot(q, [1, 2, 3])

    .. warning:: There is no check that the passed values are unit-quaternions.

    """
    q = base.getvector(q, 4)
    w = base.getvector(w, 3)
    E = q[0] * (np.eye(3, 3)) - base.skew(q[1:4])
    return 0.5 * np.r_[-np.dot(q[1:4], w), E@w]


def dotb(q, w):
    """
    Rate of change of unit-quaternion

    :arg q0: unit-quaternion
    :type q0: array_like(4)
    :arg w: 3D angular velocity in body frame
    :type w: array_like(3)
    :return: rate of change of unit quaternion
    :rtype: ndarray(4)

    ``dotb(q, w)`` is the rate of change of the elements of the unit quaternion ``q``
    which represents the orientation of a body frame with angular velocity ``w`` in
    the body frame.

    .. runblock:: pycon

        >>> from spatialmath.base import dotb, qprint
        >>> from math import sqrt
        >>> q = [1/sqrt(2), 1/sqrt(2), 0, 0]   # 90deg rotation about x-axis
        >>> dotb(q, [1, 2, 3])

    .. warning:: There is no check that the passed values are unit-quaternions.

    """
    q = base.getvector(q, 4)
    w = base.getvector(w, 3)
    E = q[0] * (np.eye(3, 3)) + base.skew(q[1:4])
    return 0.5 * np.r_[-np.dot(q[1:4], w), E@w]


def angle(q1, q2):
    """
    Angle between two unit-quaternions

    :arg q0: unit-quaternion
    :type q0: array_like(4)
    :arg q1: unit-quaternion
    :type q1: array_like(4)
    :return: angle between the rotations [radians]
    :rtype: float

    If each of the input quaternions is considered a rotated coordinate
    frame, then the angle is the smallest rotation required about a fixed
    axis, to rotate the first frame into the second.

    .. runblock:: pycon

        >>> from spatialmath.base import angle
        >>> from math import sqrt
        >>> q1 = [1/sqrt(2), 1/sqrt(2), 0, 0]    # 90deg rotation about x-axis
        >>> q2 = [1/sqrt(2), 0, 1/sqrt(2), 0]    # 90deg rotation about y-axis
        >>> angle(q1, q2)

    :References:  

    - Metrics for 3D rotations: comparison and analysis,
      Du Q. Huynh, % J.Math Imaging Vis. DOFI 10.1007/s10851-009-0161-2.

    .. warning:: There is no check that the passed values are unit-quaternions.

    """
    # TODO different methods

    q1 = base.getvector(q1, 4)
    q2 = base.getvector(q2, 4)
    return 2.0 * math.atan2(base.norm(q1 - q2), base.norm(q1 + q2))


def qprint(q, delim=('<', '>'), fmt='{: .4f}', file=sys.stdout):
    """
    Format a quaternion

    :arg q: unit-quaternion
    :type q: array_like(4)
    :arg delim: 2-list of delimeters [default ('<', '>')]
    :type delim: list or tuple of strings
    :arg fmt: printf-style format soecifier [default '{: .4f}']
    :type fmt: str
    :arg file: destination for formatted string [default sys.stdout]
    :type file: file object
    :return: formatted string
    :rtype: str

    Format the quaternion in a human-readable form as::

        S  D1  VX VY VZ D2

    where S, VX, VY, VZ are the quaternion elements, and D1 and D2 are a pair
    of delimeters given by `delim`.

    By default the string is written to `sys.stdout`.

    If `file=None` then a string is returned.

    .. runblock:: pycon

        >>> from spatialmath.base import qprint, rand
        >>> q = [1, 2, 3, 4]
        >>> qprint(q)
        >>> q = rand()   # a unit quaternion
        >>> qprint(q, delim=('<<', '>>'))
    """
    q = base.getvector(q, 4)
    template = "# {} #, #, # {}".replace('#', fmt)
    s = template.format(q[0], delim[0], q[1], q[2], q[3], delim[1])
    if file:
        file.write(s + '\n')
    else:
        return s


if __name__ == '__main__':  # pragma: no cover
    import pathlib

    exec(open(pathlib.Path(__file__).parent.parent.parent.absolute() / "tests" /
         "base" / "test_quaternions.py").read())  # pylint: disable=exec-used
