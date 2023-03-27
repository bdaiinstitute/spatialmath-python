# Part of Spatial Math Toolbox for Python
# Copyright (c) 2000 Peter Corke
# MIT Licence, see details in top-level file: LICENCE

"""
These functions create and manipulate quaternions or unit quaternions.
The quaternion is represented
by a 1D NumPy array with 4 elements: s, x, y, z.

"""
# pylint: disable=invalid-name

import sys
import math
import numpy as np
import spatialmath.base as smb
from spatialmath.base.types import *

_eps = np.finfo(np.float64).eps


def qeye() -> QuaternionArray:
    """
    Create an identity quaternion

    :return: an identity quaternion
    :rtype: ndarray(4)

    Creates an identity quaternion, with the scalar part equal to one, and
    a zero vector value.

    .. runblock:: pycon

        >>> from spatialmath.base import qeye, qprint
        >>> q = qeye()
        >>> qprint(q)

    """
    return np.r_[1, 0, 0, 0]


def qpure(v: ArrayLike3) -> QuaternionArray:
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
        >>> q = qpure([1, 2, 3])
        >>> qprint(q)
    """
    v = smb.getvector(v, 3)
    return np.r_[0, v]


def qpositive(q: ArrayLike4) -> QuaternionArray:
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


def qnorm(q: ArrayLike4) -> float:
    r"""
    Norm of a quaternion

    :arg q: quaternion
    :type v: : array_like(4)
    :return: norm of the quaternion
    :rtype: float

    Returns the norm (length or magnitude) of the input quaternion which is

    .. math::

        (s^2 + v_x^2 + v_y^2 + v_z^2)^{1/2}

    .. runblock:: pycon

        >>> from spatialmath.base import qnorm
        >>> q = qnorm([1, 2, 3, 4])
        >>> print(q)

    :seealso: :func:`qunit`

    """
    q = smb.getvector(q, 4)
    return np.linalg.norm(q)


def qunit(q: ArrayLike4, tol: Optional[float] = 10) -> UnitQuaternionArray:
    """
    Create a unit quaternion

    :arg v: quaterion
    :type v: array_like(4)
    :return: a pure quaternion
    :rtype: ndarray(4)
    :raises ValueError: quaternion has (near) zero norm

    Creates a unit quaternion, with unit norm, by scaling the input quaternion.

    .. runblock:: pycon

        >>> from spatialmath.base import qunit, qprint
        >>> q = qunit([1, 2, 3, 4])
        >>> qprint(q)

    .. note:: Scalar part is always positive.

    .. note:: If the quaternion norm is less than ``tol * eps`` an exception is
              raised.

    :seealso: :func:`qnorm`
    """
    q = smb.getvector(q, 4)
    nm = np.linalg.norm(q)
    if abs(nm) < tol * _eps:
        raise ValueError("cannot normalize (near) zero length quaternion")
    else:
        q /= nm

    if q[0] >= 0:
        return q
    else:
        return -q


def qisunit(q: ArrayLike4, tol: Optional[float] = 100) -> bool:
    """
    Test if quaternion has unit length

    :param v: quaternion
    :type v: array_like(4)
    :param tol: tolerance in units of eps
    :type tol: float
    :return: whether quaternion has unit length
    :rtype: bool

    .. runblock:: pycon

        >>> from spatialmath.base import qeye, qpure, qisunit
        >>> q = qeye()
        >>> qisunit(q)
        >>> q = qpure([1, 2, 3])
        >>> qisunit(q)

    :seealso: :func:`qunit`
    """
    return smb.iszerovec(q, tol=tol)


@overload
def qisequal(
    q1: ArrayLike4,
    q2: ArrayLike4,
    tol: Optional[float] = 100,
    unitq: Optional[bool] = False,
) -> bool:
    ...


@overload
def qisequal(
    q1: ArrayLike4,
    q2: ArrayLike4,
    tol: Optional[float] = 100,
    unitq: Optional[bool] = True,
) -> bool:
    ...


def qisequal(q1, q2, tol: Optional[float] = 100, unitq: Optional[bool] = False):
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

        >>> from spatialmath.base import qisequal
        >>> q1 = [1, 2, 3, 4]
        >>> q2 = [-1, -2, -3, -4]
        >>> qisequal(q1, q2)
        >>> qisequal(q1, q2, unitq=True)
    """
    q1 = smb.getvector(q1, 4)
    q2 = smb.getvector(q2, 4)

    if unitq:
        return (np.sum(np.abs(q1 - q2)) < tol * _eps) or (
            np.sum(np.abs(q1 + q2)) < tol * _eps
        )
    else:
        return np.sum(np.abs(q1 - q2)) < tol * _eps


def q2v(q: ArrayLike4) -> R3:
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

    :seealso: :func:`v2q`

    """
    q = smb.getvector(q, 4)
    if q[0] >= 0:
        return q[1:4]
    else:
        return -q[1:4]


def v2q(v: ArrayLike3) -> UnitQuaternionArray:
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
    v = smb.getvector(v, 3)
    s = math.sqrt(1 - np.sum(v**2))
    return np.r_[s, v]


def qqmul(q1: ArrayLike4, q2: ArrayLike4) -> QuaternionArray:
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

    :seealso: qvmul, qinner, vvmul

    """
    q1 = smb.getvector(q1, 4)
    q2 = smb.getvector(q2, 4)
    s1 = q1[0]
    v1 = q1[1:4]
    s2 = q2[0]
    v2 = q2[1:4]

    return np.r_[s1 * s2 - np.dot(v1, v2), s1 * v2 + s2 * v1 + np.cross(v1, v2)]


def qinner(q1: ArrayLike4, q2: ArrayLike4) -> float:
    """
    Quaternion inner product

    :arg q0: quaternion
    :type q0: : array_like(4)
    :arg q1: uaternion
    :type q1: array_like(4)
    :return: inner product
    :rtype: float

    This is the inner or dot product of two quaternions, it is the sum of the element-wise
    product.

    - The inner product ``inner(q, q)`` is the square of the norm of ``q``.
    - If ``q0`` and ``q1`` are unit quaternions then the inner product is the
      cosine of the angle between the two orientations.

    .. runblock:: pycon

        >>> from spatialmath.base import qinner
        >>> from math import sqrt, acos, pi
        >>> q1 = [1, 2, 3, 4]
        >>> qinner(q1, q1)                     # square of the norm
        >>> q1 = [1/sqrt(2), 1/sqrt(2), 0, 0]  # 90deg rotation about x-axis
        >>> q2 = [1/sqrt(2), 0, 1/sqrt(2), 0]  # 90deg rotation about y-axis
        >>> acos(qinner(q1, q2)) * 180 / pi    # angle between q1 and q2

    :seealso: qvmul

    """
    q1 = smb.getvector(q1, 4)
    q2 = smb.getvector(q2, 4)

    return np.dot(q1, q2)


def qvmul(q: ArrayLike4, v: ArrayLike3) -> R3:
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
    q = smb.getvector(q, 4)
    v = smb.getvector(v, 3)
    qv = qqmul(q, qqmul(qpure(v), qconj(q)))
    return qv[1:4]


def vvmul(qa: ArrayLike3, qb: ArrayLike3) -> R3:
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

    :seealso: :func:`q2v` :func:`v2q` :func:`qvmul`
    """
    t6 = math.sqrt(1.0 - np.sum(qa**2))
    t11 = math.sqrt(1.0 - np.sum(qb**2))
    return np.r_[
        qa[1] * qb[2] - qb[1] * qa[2] + qb[0] * t6 + qa[0] * t11,
        -qa[0] * qb[2] + qb[0] * qa[2] + qb[1] * t6 + qa[1] * t11,
        qa[0] * qb[1] - qb[0] * qa[1] + qb[2] * t6 + qa[2] * t11,
    ]


def qpow(q: ArrayLike4, power: int) -> QuaternionArray:
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
    q = smb.getvector(q, 4)
    if not isinstance(power, int):
        raise ValueError("Power must be an integer")
    qr = qeye()
    for _ in range(0, abs(power)):
        qr = qqmul(qr, q)

    if power < 0:
        qr = qconj(qr)

    return qr


def qconj(q: ArrayLike4) -> QuaternionArray:
    """
    Quaternion conjugate

    :arg q: quaternion
    :type v: array_like(4)
    :return: conjugate of input quaternion
    :rtype: ndarray(4)

    Conjugate of quaternion, the vector part is negated.

    .. runblock:: pycon

        >>> from spatialmath.base import qconj, qprint
        >>> q = [1, 2, 3, 4]
        >>> qprint(qconj(q))

    :SymPy: supported
    """
    q = smb.getvector(q, 4)
    return np.r_[q[0], -q[1:4]]


def q2r(
    q: Union[UnitQuaternionArray, ArrayLike4], order: Optional[str] = "sxyz"
) -> SO3Array:
    """
    Convert unit-quaternion to SO(3) rotation matrix

    :arg q: unit-quaternion
    :type v: array_like(4)
    :param order: the order of the quaternion elements. Must be 'sxyz' or
        'xyzs'. Defaults to 'sxyz'.
    :type order: str
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
    q = smb.getvector(q, 4)
    if order == "sxyz":
        s, x, y, z = q
    elif order == "xyzs":
        x, y, z, s = q
    else:
        raise ValueError("order is invalid, must be 'sxyz' or 'xyzs'")

    return np.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - s * z), 2 * (x * z + s * y)],
            [2 * (x * y + s * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - s * x)],
            [2 * (x * z - s * y), 2 * (y * z + s * x), 1 - 2 * (x**2 + y**2)],
        ]
    )


def r2q(
    R: SO3Array,
    check: Optional[bool] = False,
    tol: Optional[float] = 100,
    order: Optional[str] = "sxyz",
) -> UnitQuaternionArray:
    """
    Convert SO(3) rotation matrix to unit-quaternion

    :arg R: SO(3) rotation matrix
    :type R: ndarray(3,3)
    :param check: check validity of rotation matrix, default False
    :type check: bool
    :param tol: tolerance in units of eps
    :type tol: float
    :param order: the order of the returned quaternion elements. Must be 'sxyz' or
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
    if not smb.isrot(R, check=check, tol=tol):
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

    e = np.array(
        [
            math.sqrt(d1 + t23m + t13m + t12m) / 4.0,
            math.sqrt(t23m + d2 + t12p + t13p) / 4.0,
            math.sqrt(t13m + t12p + d3 + t23p) / 4.0,
            math.sqrt(t12m + t13p + t23p + d4) / 4.0,
        ]
    )

    i = np.argmax(e)

    if i == 0:
        e[1] = math.copysign(e[1], R[2, 1] - R[1, 2])
        e[2] = math.copysign(e[2], R[0, 2] - R[2, 0])
        e[3] = math.copysign(e[3], R[1, 0] - R[0, 1])
    elif i == 1:
        e[0] = math.copysign(e[0], R[2, 1] - R[1, 2])
        e[2] = math.copysign(e[2], R[1, 0] + R[0, 1])
        e[3] = math.copysign(e[3], R[0, 2] + R[2, 0])
    elif i == 2:
        e[0] = math.copysign(e[0], R[0, 2] - R[2, 0])
        e[1] = math.copysign(e[1], R[1, 0] + R[0, 1])
        e[3] = math.copysign(e[3], R[2, 1] + R[1, 2])
    else:
        e[0] = math.copysign(e[0], R[1, 0] - R[0, 1])
        e[1] = math.copysign(e[1], R[0, 2] + R[2, 0])
        e[2] = math.copysign(e[2], R[2, 1] + R[1, 2])

    if order == "sxyz":
        return e
    elif order == "xyzs":
        return e[[1, 2, 3, 0]]
    else:
        raise ValueError("order is invalid, must be 'sxyz' or 'xyzs'")


# def r2q_svd(R):
#     U = np.array(
#         [
#             [
#                 R[0, 0] + R[1, 1] + R[2, 2] + 1,
#                 R[2, 1] - R[1, 2],
#                 -R[2, 0] + R[0, 2],
#                 R[1, 0] - R[0, 1],
#             ],
#             [
#                 R[2, 1] - R[1, 2],
#                 R[0, 0] - R[1, 1] - R[2, 2] + 1,
#                 R[1, 0] + R[0, 1],
#                 R[2, 0] + R[0, 2],
#             ],
#             [
#                 -R[2, 0] + R[0, 2],
#                 R[1, 0] + R[0, 1],
#                 -R[0, 0] + R[1, 1] - R[2, 2] + 1,
#                 R[2, 1] + R[1, 2],
#             ],
#             [
#                 R[1, 0] - R[0, 1],
#                 R[2, 0] + R[0, 2],
#                 R[2, 1] + R[1, 2],
#                 -R[0, 0] - R[1, 1] + R[2, 2] + 1,
#             ],
#         ]
#     )

#     U, S, VT = np.linalg.svd(U)

#     e = U[:, 0]
#     # if e[0] < -10 * _eps:
#     #     e = -e
#     return e


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

#     :reference:
#         - Funda, Taylor, IEEE Trans. Robotics and Automation, 6(3),
#           June 1990, pp.382-388.  (coding reference)
#         - Sarabandi, S., and Thomas, F. (March 1, 2019).
#           "A Survey on the Computation of Quaternions From Rotation Matrices."
#           ASME. J. Mechanisms Robotics. April 2019; 11(2): 021006. (according to this
#           paper the algorithm is Hughes' method)


#     :seealso: :func:`q2r`
#     """
#     if not smb.isrot(R, check=check, tol=tol):
#         raise ValueError("Argument must be a valid SO(3) matrix")

#     qs = math.sqrt(max(0, np.trace(R) + 1)) / 2.0  # scalar part
#     kx = R[2, 1] - R[1, 2]  # Oz - Ay
#     ky = R[0, 2] - R[2, 0]  # Ax - Nz
#     kz = R[1, 0] - R[0, 1]  # Ny - Ox

#     # equation (7)
#     if (R[0, 0] >= R[1, 1]) and (R[0, 0] >= R[2, 2]):
#         kx1 = R[0, 0] - R[1, 1] - R[2, 2] + 1  # Nx - Oy - Az + 1
#         ky1 = R[1, 0] + R[0, 1]  # Ny + Ox
#         kz1 = R[2, 0] + R[0, 2]  # Nz + Ax
#         add = kx >= 0
#     elif R[1, 1] >= R[2, 2]:
#         kx1 = R[1, 0] + R[0, 1]  # Ny + Ox
#         ky1 = R[1, 1] - R[0, 0] - R[2, 2] + 1  # Oy - Nx - Az + 1
#         kz1 = R[2, 1] + R[1, 2]  # Oz + Ay
#         add = ky >= 0
#     else:
#         kx1 = R[2, 0] + R[0, 2]  # Nz + Ax
#         ky1 = R[2, 1] + R[1, 2]  # Oz + Ay
#         kz1 = R[2, 2] - R[0, 0] - R[1, 1] + 1  # Az - Nx - Oy + 1
#         add = kz >= 0

#     # equation (8)
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
#         return qeye()
#     else:
#         return np.r_[qs, (math.sqrt(1.0 - qs**2) / nm) * kv]


def qslerp(
    q0: ArrayLike4, q1: ArrayLike4, s: float, shortest: Optional[bool] = False
) -> UnitQuaternionArray:
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

        >>> from spatialmath.base import qslerp, qprint
        >>> from math import sqrt
        >>> q0 = [1/sqrt(2), 1/sqrt(2), 0, 0]  # 90deg rotation about x-axis
        >>> q1 = [1/sqrt(2), 0, 1/sqrt(2), 0]  # 90deg rotation about y-axis
        >>> qprint(qslerp(q0, q1, 0))           # this is q0
        >>> qprint(qslerp(q0, q1, 1))           # this is q1
        >>> qprint(qslerp(q0, q1, 0.5))         # this is in "half way" between

    .. warning:: There is no check that the passed values are unit-quaternions.

    """
    if not 0 <= s <= 1:
        raise ValueError("s must be in the interval [0,1]")
    q0 = smb.getvector(q0, 4)
    q1 = smb.getvector(q1, 4)

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
            q0 = -q0  # pylint: disable=invalid-unary-operand-type
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


def qrand() -> UnitQuaternionArray:
    """
    Random unit-quaternion

    :return: random unit-quaternion
    :rtype: ndarray(4)

    Computes a uniformly distributed random unit-quaternion which can be
    considered equivalent to a random SO(3) rotation.

    .. runblock:: pycon

        >>> from spatialmath.base import qrand, qprint
        >>> qprint(qrand())
    """
    u = np.random.uniform(low=0, high=1, size=3)  # get 3 random numbers in [0,1]
    return np.r_[
        math.sqrt(1 - u[0]) * math.sin(2 * math.pi * u[1]),
        math.sqrt(1 - u[0]) * math.cos(2 * math.pi * u[1]),
        math.sqrt(u[0]) * math.sin(2 * math.pi * u[2]),
        math.sqrt(u[0]) * math.cos(2 * math.pi * u[2]),
    ]


def qmatrix(q: ArrayLike4) -> R4x4:
    """
    Convert quaternion to 4x4 matrix equivalent

    :arg q: quaternion
    :type v: array_like(4)
    :return: equivalent matrix
    :rtype: ndarray(4)

    Hamilton multiplication between two quaternions can be considered as a
    matrix-vector product, the left-hand quaternion is represented by an
    equivalent 4x4 matrix and the right-hand quaternion as 4x1 column vector.

    .. runblock:: pycon

        >>> from spatialmath.base import qmatrix, qqmul, qprint
        >>> q1 = [1, 2, 3, 4]
        >>> q2 = [5, 6, 7, 8]
        >>> qqmul(q1, q2)    # conventional Hamilton product
        >>> m = qmatrix(q1)
        >>> print(m)
        >>> v = m @ np.array(q2)
        >>> print(v)

    :seealso: qqmul

    """
    q = smb.getvector(q, 4)
    s = q[0]
    x = q[1]
    y = q[2]
    z = q[3]
    return np.array([[s, -x, -y, -z], [x, s, -z, y], [y, z, s, -x], [z, -y, x, s]])


def qdot(q: ArrayLike4, w: ArrayLike3) -> QuaternionArray:
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

        >>> from spatialmath.base import qdot, qprint
        >>> from math import sqrt
        >>> q = [1/sqrt(2), 1/sqrt(2), 0, 0]   # 90deg rotation about x-axis
        >>> qdot(q, [1, 2, 3])

    .. warning:: There is no check that the passed values are unit-quaternions.

    """
    q = smb.getvector(q, 4)
    w = smb.getvector(w, 3)
    E = q[0] * (np.eye(3, 3)) - smb.skew(q[1:4])
    return 0.5 * np.r_[-np.dot(q[1:4], w), E @ w]


def qdotb(q: ArrayLike4, w: ArrayLike3) -> QuaternionArray:
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

        >>> from spatialmath.base import qdotb, qprint
        >>> from math import sqrt
        >>> q = [1/sqrt(2), 1/sqrt(2), 0, 0]   # 90deg rotation about x-axis
        >>> qdotb(q, [1, 2, 3])

    .. warning:: There is no check that the passed values are unit-quaternions.

    """
    q = smb.getvector(q, 4)
    w = smb.getvector(w, 3)
    E = q[0] * (np.eye(3, 3)) + smb.skew(q[1:4])
    return 0.5 * np.r_[-np.dot(q[1:4], w), E @ w]


def qangle(q1: ArrayLike4, q2: ArrayLike4) -> float:
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

        >>> from spatialmath.base import qangle
        >>> from math import sqrt
        >>> q1 = [1/sqrt(2), 1/sqrt(2), 0, 0]    # 90deg rotation about x-axis
        >>> q2 = [1/sqrt(2), 0, 1/sqrt(2), 0]    # 90deg rotation about y-axis
        >>> qangle(q1, q2)

    :References:

    - Metrics for 3D rotations: comparison and analysis,
      Du Q. Huynh, % J.Math Imaging Vis. DOFI 10.1007/s10851-009-0161-2.

    .. warning:: There is no check that the passed values are unit-quaternions.

    """
    # TODO different methods

    q1 = smb.getvector(q1, 4)
    q2 = smb.getvector(q2, 4)
    return 2.0 * math.atan2(smb.norm(q1 - q2), smb.norm(q1 + q2))


def qprint(
    q: Union[ArrayLike4, ArrayLike4],
    delim: Optional[Tuple[str, str]] = ("<", ">"),
    fmt: Optional[str] = "{: .4f}",
    file: Optional[TextIO] = sys.stdout,
) -> str:
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

        >>> from spatialmath.base import qprint, qrand
        >>> q = [1, 2, 3, 4]
        >>> qprint(q)
        >>> q = qrand()   # a unit quaternion
        >>> qprint(q, delim=('<<', '>>'))
    """
    q = smb.getvector(q, 4)
    template = "# {} #, #, # {}".replace("#", fmt)
    s = template.format(q[0], delim[0], q[1], q[2], q[3], delim[1])
    if file:
        file.write(s + "\n")
    else:
        return s


if __name__ == "__main__":  # pragma: no cover
    import pathlib

    exec(
        open(
            pathlib.Path(__file__).parent.parent.parent.absolute()
            / "tests"
            / "base"
            / "test_quaternions.py"
        ).read()
    )  # pylint: disable=exec-used
