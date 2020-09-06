#!/usr/bin/env python3

# pylint: disable=invalid-name

import sys
import math
import numpy as np
from spatialmath import base as tr
from spatialmath.base import argcheck

_eps = np.finfo(np.float64).eps


def eye():
    """
    Create an identity quaternion

    :return: an identity quaternion
    :rtype: numpy.ndarray, shape=(4,)

    Creates an identity quaternion, with the scalar part equal to one, and
    a zero vector value.

    """
    return np.r_[1, 0, 0, 0]


def pure(v):
    """
    Create a pure quaternion

    :arg v: vector from a 3-vector
    :type v: array_like
    :return: pure quaternion
    :rtype: numpy.ndarray, shape=(4,)

    Creates a pure quaternion, with a zero scalar value and the vector part
    equal to the passed vector value.

    """
    v = argcheck.getvector(v, 3)
    return np.r_[0, v]


def qnorm(q):
    r"""
    Norm of a quaternion

    :arg q: input quaternion as a 4-vector
    :type v: : array_like
    :return: norm of the quaternion
    :rtype: float

    Returns the norm, length or magnitude of the input quaternion which is
    :math:`\sqrt{s^2 + v_x^2 + v_y^2 + v_z^2}`

    :seealso: unit

    """
    q = argcheck.getvector(q, 4)
    return np.linalg.norm(q)


def unit(q, tol=10):
    """
    Create a unit quaternion

    :arg v: quaterion as a 4-vector
    :type v: array_like
    :return: a pure quaternion
    :rtype: numpy.ndarray, shape=(4,)

    Creates a unit quaternion, with unit norm, by scaling the input quaternion.

    .. seealso:: norm
    """
    q = argcheck.getvector(q, 4)
    nm = np.linalg.norm(q)
    assert abs(nm) > tol * _eps, 'cannot normalize (near) zero length quaternion'
    return q / nm


def isunit(q, tol=10):
    """
    Test if quaternion has unit length

    :param v: quaternion as a 4-vector
    :type v: array_like
    :param tol: tolerance in units of eps
    :type tol: float
    :return: whether quaternion has unit length
    :rtype: bool

    :seealso: unit
    """
    return tr.iszerovec(q, tol=tol)


def isequal(q1, q2, tol=100, unitq=False):
    """
    Test if quaternions are equal

    :param q1: quaternion as a 4-vector
    :type q1: array_like
    :param q2: quaternion as a 4-vector
    :type q2: array_like
    :param unitq: quaternions are unit quaternions
    :type unitq: bool
    :param tol: tolerance in units of eps
    :type tol: float
    :return: whether quaternion has unit length
    :rtype: bool

    Tests if two quaternions are equal.

    For unit-quaternions ``unitq=True`` the double mapping is taken into account,
    that is ``q`` and ``-q`` represent the same orientation and ``isequal(q, -q, unitq=True)`` will
    return ``True``.
    """
    q1 = argcheck.getvector(q1, 4)
    q2 = argcheck.getvector(q2, 4)

    if unitq:
        return (np.sum(np.abs(q1 - q2)) < tol * _eps) or (np.sum(np.abs(q1 + q2)) < tol * _eps)
    else:
        return np.sum(np.abs(q1 - q2)) < tol * _eps


def q2v(q):
    """
    Convert unit-quaternion to 3-vector

    :arg q: unit-quaternion as a 4-vector
    :type v: array_like
    :return: a unique 3-vector
    :rtype: numpy.ndarray, shape=(3,)

    Returns a unique 3-vector representing the input unit-quaternion. The sign
    of the scalar part is made positive, if necessary by multiplying the
    entire quaternion by -1, then the vector part is taken.

    .. warning:: There is no check that the passed value is a unit-quaternion.

    .. seealso:: v2q

    """
    q = argcheck.getvector(q, 4)
    if q[0] >= 0:
        return q[1:4]
    else:
        return -q[1:4]


def v2q(v):
    r"""
    Convert 3-vector to unit-quaternion

    :arg v: vector part of unit quaternion, a 3-vector
    :type v: array_like
    :return: a unit quaternion
    :rtype: numpy.ndarray, shape=(4,)

    Returns a unit-quaternion reconsituted from just its vector part.  Assumes
    that the scalar part was positive, so :math:`s = \sqrt{1-||v||}`.

    .. seealso:: q2v
    """
    v = argcheck.getvector(v, 3)
    s = math.sqrt(1 - np.sum(v**2))
    return np.r_[s, v]


def qqmul(q1, q2):
    """
    Quaternion multiplication

    :arg q0: left-hand quaternion as a 4-vector
    :type q0: : array_like
    :arg q1: right-hand quaternion as a 4-vector
    :type q1: array_like
    :return: quaternion product
    :rtype: numpy.ndarray, shape=(4,)

    This is the quaternion or Hamilton product.  If both operands are unit-quaternions then
    the product will be a unit-quaternion.

    :seealso: qvmul, inner, vvmul

    """
    q1 = argcheck.getvector(q1, 4)
    q2 = argcheck.getvector(q2, 4)
    s1 = q1[0]
    v1 = q1[1:4]
    s2 = q2[0]
    v2 = q2[1:4]

    return np.r_[s1 * s2 - np.dot(v1, v2), s1 * v2 + s2 * v1 + np.cross(v1, v2)]


def inner(q1, q2):
    """
    Quaternion innert product

    :arg q0: quaternion as a 4-vector
    :type q0: : array_like
    :arg q1: uaternion as a 4-vector
    :type q1: array_like
    :return: inner product
    :rtype: numpy.ndarray, shape=(4,)

    This is the inner or dot product of two quaternions, it is the sum of the element-wise
    product.

    :seealso: qvmul

    """
    q1 = argcheck.getvector(q1, 4)
    q2 = argcheck.getvector(q2, 4)

    return np.dot(q1, q2)


def qvmul(q, v):
    """
    Vector rotation

    :arg q: unit-quaternion as a 4-vector
    :type q: array_like
    :arg v: 3-vector to be rotated
    :type v: list, tuple, numpy.ndarray
    :return: rotated 3-vector
    :rtype: numpy.ndarray, shape=(3,)

    The vector `v` is rotated about the origin by the SO(3) equivalent of the unit
    quaternion.

    .. warning:: There is no check that the passed value is a unit-quaternions.

    :seealso: qvmul
    """
    q = argcheck.getvector(q, 4)
    v = argcheck.getvector(v, 3)
    qv = qqmul(q, qqmul(pure(v), conj(q)))
    return qv[1:4]


def vvmul(qa, qb):
    """
    Quaternion multiplication


    :arg qa: left-hand quaternion as a 3-vector
    :type qa: : array_like
    :arg qb: right-hand quaternion as a 3-vector
    :type qb: array_like
    :return: quaternion product
    :rtype: numpy.ndarray, shape=(3,)

    This is the quaternion or Hamilton product of unit-quaternions defined only
    by their vector components.  The product will be a unit-quaternion, defined only
    by its vector component.

    :seealso: qvmul, inner
    """
    t6 = math.sqrt(1.0 - np.sum(qa**2))
    t11 = math.sqrt(1.0 - np.sum(qb**2))
    return np.r_[qa[1] * qb[2] - qb[1] * qa[2] + qb[0] * t6 + qa[0] * t11, -qa[0] * qb[2] + qb[0] * qa[2] + qb[1] * t6 + qa[1] * t11, qa[0] * qb[1] - qb[0] * qa[1] + qb[2] * t6 + qa[2] * t11]


def qpow(q, power):
    """
    Raise quaternion to a power

    :arg q: quaternion as a 4-vector
    :type v: array_like
    :arg power: exponent
    :type power: int
    :return: input quaternion raised to the specified power
    :rtype: numpy.ndarray, shape=(4,)

    Raises a quaternion to the specified power using repeated multiplication.

    Notes:

    - power must be an integer
    - power can be negative, in which case the conjugate is taken

    """
    q = argcheck.getvector(q, 4)
    assert isinstance(power, int), "Power must be an integer"
    qr = eye()
    for _ in range(0, abs(power)):
        qr = qqmul(qr, q)

    if power < 0:
        qr = conj(qr)

    return qr


def conj(q):
    """
    Quaternion conjugate

    :arg q: quaternion as a 4-vector
    :type v: array_like
    :return: conjugate of input quaternion
    :rtype: numpy.ndarray, shape=(4,)

    Conjugate of quaternion, the vector part is negated.

    """
    q = argcheck.getvector(q, 4)
    return np.r_[q[0], -q[1:4]]


def q2r(q):
    """
    Convert unit-quaternion to SO(3) rotation matrix

    :arg q: unit-quaternion as a 4-vector
    :type v: array_like
    :return: corresponding SO(3) rotation matrix
    :rtype: numpy.ndarray, shape=(3,3)

    Returns an SO(3) rotation matrix corresponding to this unit-quaternion.

    .. warning:: There is no check that the passed value is a unit-quaternion.

    :seealso: r2q

    """
    q = argcheck.getvector(q, 4)
    s = q[0]
    x = q[1]
    y = q[2]
    z = q[3]
    return np.array([[1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - s * z), 2 * (x * z + s * y)],
                     [2 * (x * y + s * z), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - s * x)],
                     [2 * (x * z - s * y), 2 * (y * z + s * x), 1 - 2 * (x ** 2 + y ** 2)]])


def r2q(R, check=False, tol=100):
    """
    Convert SO(3) rotation matrix to unit-quaternion

    :arg R: rotation matrix
    :type R: numpy.ndarray, shape=(3,3)
    :param check: check validity of rotation matrix, default False
    :type check: bool
    :param tol: tolerance in units of eps
    :type tol: float
    :return: unit-quaternion
    :rtype: numpy.ndarray, shape=(3,)

    Returns a unit-quaternion corresponding to the input SO(3) rotation matrix.

    .. warning:: There is no check that the passed matrix is a valid rotation matrix.

    :seealso: q2r

    """
    assert R.shape == (3, 3) and tr.isR(R), "Argument must be 3x3 rotation matrix"
    if check:
        assert tr.isR(R, tol=tol), "Argument must be a valid SO(3) matrix"

    qs = math.sqrt(max(0, np.trace(R) + 1)) / 2.0
    kx = R[2, 1] - R[1, 2]  # Oz - Ay
    ky = R[0, 2] - R[2, 0]  # Ax - Nz
    kz = R[1, 0] - R[0, 1]  # Ny - Ox

    if (R[0, 0] >= R[1, 1]) and (R[0, 0] >= R[2, 2]):
        kx1 = R[0, 0] - R[1, 1] - R[2, 2] + 1  # Nx - Oy - Az + 1
        ky1 = R[1, 0] + R[0, 1]  # Ny + Ox
        kz1 = R[2, 0] + R[0, 2]  # Nz + Ax
        add = (kx >= 0)
    elif R[1, 1] >= R[2, 2]:
        kx1 = R[1, 0] + R[0, 1]  # Ny + Ox
        ky1 = R[1, 1] - R[0, 0] - R[2, 2] + 1  # Oy - Nx - Az + 1
        kz1 = R[2, 1] + R[1, 2]  # Oz + Ay
        add = (ky >= 0)
    else:
        kx1 = R[2, 0] + R[0, 2]  # Nz + Ax
        ky1 = R[2, 1] + R[1, 2]  # Oz + Ay
        kz1 = R[2, 2] - R[0, 0] - R[1, 1] + 1  # Az - Nx - Oy + 1
        add = (kz >= 0)

    if add:
        kx = kx + kx1
        ky = ky + ky1
        kz = kz + kz1
    else:
        kx = kx - kx1
        ky = ky - ky1
        kz = kz - kz1

    kv = np.r_[kx, ky, kz]
    nm = np.linalg.norm(kv)
    if abs(nm) < tol * _eps:
        return eye()
    else:
        return np.r_[qs, (math.sqrt(1.0 - qs ** 2) / nm) * kv]


def slerp(q0, q1, s, shortest=False):
    """
    Quaternion conjugate

    :arg q0: initial unit quaternion as a 4-vector
    :type q0: array_like
    :arg q1: final unit quaternion as a 4-vector
    :type q1: array_like
    :arg s: interpolation coefficient in the range [0,1]
    :type s: float
    :arg shortest: choose shortest distance [default False]
    :type shortest: bool
    :return: interpolated unit-quaternion
    :rtype: numpy.ndarray, shape=(4,)

    An interpolated quaternion between ``q0`` when ``s`` = 0 to ``q1`` when ``s`` = 1.

    Interpolation is performed on a great circle on a 4D hypersphere. This is
    a rotation about a single fixed axis in space which yields the straightest
    and shortest path between two points.

    For large rotations the path may be the *long way around* the circle,
    the option ``'shortest'`` ensures always the shortest path.

    .. warning:: There is no check that the passed values are unit-quaternions.

    """
    assert 0 <= s <= 1, 's must be in the interval [0,1]'
    q0 = argcheck.getvector(q0, 4)
    q1 = argcheck.getvector(q1, 4)

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
            dotprod = -dotprod # pylint: disable=invalid-unary-operand-type

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
    :rtype: numpy.ndarray, shape=(4,)

    Computes a uniformly distributed random unit-quaternion which can be
    considered equivalent to a random SO(3) rotation.
    """
    u = np.random.uniform(low=0, high=1, size=3)  # get 3 random numbers in [0,1]
    return np.r_[
        math.sqrt(1 - u[0]) * math.sin(2 * math.pi * u[1]),
        math.sqrt(1 - u[0]) * math.cos(2 * math.pi * u[1]),
        math.sqrt(u[0]) * math.sin(2 * math.pi * u[2]),
        math.sqrt(u[0]) * math.cos(2 * math.pi * u[2])]


def matrix(q):
    """
    Convert to 4x4 matrix equivalent

    :arg q: quaternion as a 4-vector
    :type v: array_like
    :return: equivalent matrix
    :rtype: numpy.ndarray, shape=(4,4)

    Hamilton multiplication between two quaternions can be considered as a
    matrix-vector product, the left-hand quaternion is represented by an
    equivalent 4x4 matrix and the right-hand quaternion as 4x1 column vector.

    :seealso: qqmul

    """
    q = argcheck.getvector(q, 4)
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

    :arg q0: unit-quaternion as a 4-vector
    :type q0: array_like
    :arg w: angular velocity in world frame as a 3-vector
    :type w: array_like
    :return: rate of change of unit quaternion
    :rtype: numpy.ndarray, shape=(4,)

    ``dot(q, w)`` is the rate of change of the elements of the unit quaternion ``q``
    which represents the orientation of a body frame with angular velocity ``w`` in
    the world frame.

    .. warning:: There is no check that the passed values are unit-quaternions.

    """
    q = argcheck.getvector(q, 4)
    w = argcheck.getvector(w, 3)
    E = q[0] * (np.eye(3, 3)) - tr.skew(q[1:4])
    return 0.5 * np.r_[-np.dot(q[1:4], w), E@w]


def dotb(q, w):
    """
    Rate of change of unit-quaternion

    :arg q0: unit-quaternion as a 4-vector
    :type q0: array_like
    :arg w: angular velocity in body frame as a 3-vector
    :type w: array_like
    :return: rate of change of unit quaternion
    :rtype: numpy.ndarray, shape=(4,)

    ``dotb(q, w)`` is the rate of change of the elements of the unit quaternion ``q``
    which represents the orientation of a body frame with angular velocity ``w`` in
    the body frame.

    .. warning:: There is no check that the passed values are unit-quaternions.

    """
    q = argcheck.getvector(q, 4)
    w = argcheck.getvector(w, 3)
    E = q[0] * (np.eye(3, 3)) + tr.skew(q[1:4])
    return 0.5 * np.r_[-np.dot(q[1:4], w), E@w]


def angle(q1, q2):
    """
    Angle between two unit-quaternions

    :arg q0: unit-quaternion as a 4-vector
    :type q0: array_like
    :arg q1: unit-quaternion as a 4-vector
    :type q1: array_like
    :return: angle between the rotations [radians]
    :rtype: float

    If each of the input quaternions is considered a rotated coordinate
    frame, then the angle is the smallest rotation required about a fixed
    axis, to rotate the first frame into the second.

    References:  Metrics for 3D rotations: comparison and analysis,
    Du Q. Huynh, % J.Math Imaging Vis. DOFI 10.1007/s10851-009-0161-2.

    .. warning:: There is no check that the passed values are unit-quaternions.

    """
    # TODO different methods

    q1 = argcheck.getvector(q1, 4)
    q2 = argcheck.getvector(q2, 4)
    return 2.0 * math.atan2(tr.norm(q1 - q2), tr.norm(q1 + q2))


def qprint(q, delim=('<', '>'), fmt='%f', file=sys.stdout):
    """
    Format a quaternion

    :arg q: unit-quaternion as a 4-vector
    :type q: array_like
    :arg delim: 2-list of delimeters [default ('<', '>')]
    :type delim: list or tuple of strings
    :arg fmt: printf-style format soecifier [default '%f']
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

    """
    q = argcheck.getvector(q, 4)
    template = "# %s #, #, # %s".replace('#', fmt)
    s = template % (q[0], delim[0], q[1], q[2], q[3], delim[1])
    if file:
        file.write(s + '\n')
    else:
        return s


if __name__ == '__main__':  # pragma: no cover
    import pathlib

    exec(open(pathlib.Path(__file__).parent.absolute() / "test_quaternions.py").read())  # pylint: disable=exec-used
