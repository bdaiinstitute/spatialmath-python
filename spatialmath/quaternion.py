"""
Classes to abstract quaternions and unit-quaternions.

To use::

    from spatialmath.quaternion import *
    T = UnitQuaternion.Rx(0.3)

    import spatialmath as sm
    T = sm.UnitQuaternion.Rx(0.3)

 .. inheritance-diagram:: spatialmath.quaternion
    :top-classes: collections.UserList
    :parts: 1
"""
# pylint: disable=invalid-name

import math
import numpy as np
from typing import Any
from spatialmath import base as tr
from spatialmath.base import quaternions as quat
from spatialmath.base import argcheck
from spatialmath.pose3d import SO3, SE3
from spatialmath.smuserlist import SMUserList

class Quaternion(SMUserList):
    r"""
    A quaternion is a compact method of representing a 3D rotation that has
    computational advantages including speed and numerical robustness.

    A quaternion has 2 parts, a scalar :math:`s`, and a 3-vector :math:`v` and
    is typically written as

    :math:`q = s \langle v_x, v_y, v_z \rangle`

    .. inheritance-diagram:: spatialmath.quaternion.Quaternion
       :top-classes: collections.UserList
       :parts: 1
    """

    def __init__(self, s: Any = None, v=None, check=True):
        r"""
        Construct a new quaternion

        :param s: scalar
        :type s: float
        :param v: vector
        :type v: 3-element array_like

        - ``Quaternion()`` constructs a zero quaternion
        - ``Quaternion(s, v)`` construct a new quaternion from the scalar ``s``
          and the vector ``v``
        - ``Quaternion(q)`` construct a new quaternion from the 4-vector
          ``q = [s, v]``
        - ``Quaternion([q1, q2 .. qN])`` construct a new quaternion with ``N``
          values where each element is a 4-vector
        - ``Quaternion([Q1, Q2 .. QN])`` construct a new quaternion with ``N``
          values where each element is a Quaternion instance
        - ``Quaternion(M)`` construct a new quaternion with ``N`` values where
          ``Q`` is a 4xN NumPy array.

        Examples::

            >>> Quaternion()
            0.000000 < 0.000000, 0.000000, 0.000000 >
            >>> Quaternion(1, [2,3,4])
            1.000000 < 2.000000, 3.000000, 4.000000 >
            >>> Quaternion([1,2,3,4])
            1.000000 < 2.000000, 3.000000, 4.000000 >

            >>> q=Quaternion([np.r_[1,2,3,4], np.r_[5,6,7,8]])
            >>> len(q)
            2
            >>> q
            1.000000 < 2.000000, 3.000000, 4.000000 >
            5.000000 < 6.000000, 7.000000, 8.000000 >
        """
        super().__init__()

        if v is None:
            # single argument
            if super().arghandler(s, check=False):
                return

            elif argcheck.isvector(s, 4):
                self.data = [argcheck.getvector(s)]

        elif argcheck.isscalar(s) and argcheck.isvector(v, 3):
            # Quaternion(s, v)
            self.data = [np.r_[s, argcheck.getvector(v)]]

        else:
            raise ValueError('bad argument to Quaternion constructor')

    @staticmethod
    def _identity():
        return np.zeros((4,))

    @property
    def shape(self):
        """
        Shape of the object's interal matrix representation

        :return: (4,)
        :rtype: tuple
        """
        return (4,)

    @staticmethod
    def isvalid(x):
        """
        Test if matrix is valid quaternion

        :param x: matrix to test
        :type x: numpy.ndarray
        :return: true of the matrix is 4x1.
        :rtype: bool
        """
        return x.shape == (4,)

    @property
    def s(self):
        """
        Scalar part of quaternion

        :return: scalar part of quaternion
        :rtype: float or numpy.ndarray

        ``q.s`` is the scalar part.  If `len(q)` is:

            - 1, return a scalar float
            - N>1, return a NumPy array shape=(N,) is returned.

        Examples::

            >>> Quaternion([1,2,3,4]).s
            1.0
            >>> Quaternion([np.r_[1,2,3,4], np.r_[5,6,7,8]]).s
            array([1., 5.])

        """
        if len(self) == 1:
            return self._A[0]
        else:
            return np.array([q.s for q in self])

    @property
    def v(self):
        """
        Vector part of quaternion

        :return: vector part of quaternion
        :rtype: NumPy ndarray

        ``q.v`` is the vector part.  If `len(q)` is:

            - 1, return a NumPy array shape=(3,)
            - N>1, return a NumPy array shape=(N,3).

        Examples::

            >>> Quaternion([1,2,3,4]).v
            array([2., 3., 4.])
            >>> Quaternion([np.r_[1,2,3,4], np.r_[5,6,7,8]]).v
            array([[2., 3., 4.],
                [6., 7., 8.]])
        """
        if len(self) == 1:
            return self._A[1:4]
        else:
            return np.array([q.v for q in self])

    @property
    def vec(self):
        """
        Quaternion as a vector

        :return: quaternion expressed as a 4-vector
        :rtype: numpy ndarray, shape=(4,)

        ``q.vec`` is the quaternion as a vector.  If `len(q)` is:

            - 1, return a NumPy array shape=(4,)
            - N>1, return a NumPy array shape=(N,4).

        Examples::

            >>> Quaternion([1,2,3,4]).vec
            array([1., 2., 3., 4.])
            >>> Quaternion([np.r_[1,2,3,4], np.r_[5,6,7,8]]).vec
            array([[1., 2., 3., 4.],
                [5., 6., 7., 8.]])
        """
        if len(self) == 1:
            return self._A
        else:
            return np.array([q._A for q in self])

    @classmethod
    def Pure(cls, v):
        r"""
        Construct a pure quaternion from a vector

        :param v: vector
        :type v: 3-element array_like

        ``Quaternion.Pure(v)`` is a Quaternion with a zero scalar part and the
        vector part set to ``v``,
        ie. :math:`q = 0 \langle v_x, v_y, v_z \rangle`

        Examples::

            >>> Quaternion.pure([1,2,3])
            0.000000 < 1.000000, 2.000000, 3.000000 >
        """
        return cls(s=0, v=argcheck.getvector(v, 3))

    def conj(self):
        r"""
        Conjugate of quaternion

        :rtype: Quaternion instance

        ``q.conj()`` is the quaternion ``q`` with the vector part negated, ie.
        :math:`q = s \langle -v_x, -v_y, -v_z \rangle`

        Examples::

            >>> Quaternion.pure([1,2,3]).conj()
            0.000000 < -1.000000, -2.000000, -3.000000 >
        """

        return self.__class__([quat.conj(q._A) for q in self])

    def norm(self):
        r"""
        Norm of quaternion

        :rtype: float

        ``q.norm()`` is the norm or length of the quaternion and is equal to
        :math:`\sqrt{s^2 + v_x^2 + v_y^2 + v_z^2}`


        Examples::

            >>> Quaternion([1,2,3,4]).norm()
            5.477225575051661
            >>> Quaternion([np.r_[1,2,3,4], np.r_[5,6,7,8]]).norm()
            array([ 5.47722558, 13.19090596])
        """
        if len(self) == 1:
            return quat.qnorm(self._A)
        else:
            return np.array([quat.qnorm(q._A) for q in self])

    def unit(self):
        r"""
        Unit quaternion

        :rtype: UnitQuaternion instance

        ``q.unit()`` is the quaternion ``q`` normalized to have a unit length.

        Examples::

            >>> Quaternion([1,2,3,4]).unit()
            0.182574 << 0.365148, 0.547723, 0.730297 >>
            >>> Quaternion([np.r_[1,2,3,4], np.r_[5,6,7,8]]).unit()
            0.182574 << 0.365148, 0.547723, 0.730297 >>
            0.379049 << 0.454859, 0.530669, 0.606478 >>

        Note that the return type is different, a ``UnitQuaternion``, which is
        distinguished by the use of double angle brackets to delimit the 
        vector part.
        """
        return UnitQuaternion([quat.unit(q._A) for q in self], norm=False)

    @property
    def matrix(self):
        """
        Matrix equivalent of quaternion

        :rtype: Numpy array, shape=(4,4)

        ``q.matrix`` is a 4x4 matrix which encodes the arithmetic rules of Hamilton multiplication.
        This matrix, multiplied by the 4-vector equivalent of a second quaternion, results in the 4-vector
        equivalent of the Hamilton product.

        Examples::

            >>> Quaternion([1,2,3,4]).matrix
            array([[ 1., -2., -3., -4.],
            [ 2.,  1., -4.,  3.],
            [ 3.,  4.,  1., -2.],
            [ 4., -3.,  2.,  1.]])

            # Hamilton product
            >>> Quaternion([1,2,3,4]) * Quaternion([5,6,7,8])  
            -60.000000 < 12.000000, 30.000000, 24.000000 >

            # matrix-vector product
            >>> Quaternion([1,2,3,4]).matrix @ Quaternion([5,6,7,8]).vec  
            array([-60.,  12.,  30.,  24.])
        """

        return quat.matrix(self._A)

    #-------------------------------------------- arithmetic

    def inner(self, other):
        """
        Innert product of quaternions

        :rtype: float

        ``q1.inner(q2)`` is the dot product of the equivalent vectors, 
        ie. ``numpy.dot(q1.vec, q2.vec)``.
        The value of ``q.inner(q)`` is the same as ``q.norm ** 2``.

        Examples::

            >>> Quaternion([1,2,3,4]).inner(Quaternion([5,6,7,8]))
            70.0
            >>> numpy.dot([1,2,3,4], [5,6,7,8])
            70
        """

        assert isinstance(other, Quaternion), \
            'operands to inner must be Quaternion subclass'
        return self.binop(other, quat.inner, list1=False)

    def __eq__(left, right):
        """
        Overloaded ``==`` operator

        :rtype: bool

        ``q1 == q2`` is True if ``q1` is elementwise equal to ``q2``.

        Examples::

            >>> q1 = Quaternion([1,2,3,4])
            >>> q2 = Quaternion([5,6,7,8])
            >>> q1 == q1
            True
            >>> q1 == q2
            False
            >>> Quaternion([np.r_[1,2,3,4], np.r_[5,6,7,8]]) == q1
            [True, False]
            >>> Quaternion([np.r_[1,2,3,4], np.r_[5,6,7,8]]) == q2
            [False, True]
            >>> Quaternion([np.r_[1,2,3,4], np.r_[5,6,7,8]]) == Quaternion([np.r_[1,2,3,4], np.r_[5,6,7,8]])
            [True, True]

        :seealso: :func:`__ne__`, :func:`spatialmath.base.quaternions.isequal`
        """
        assert isinstance(left, type(right)), \
            'operands to == are of different types'
        return left.binop(right, quat.isequal, list1=False)

    def __ne__(left, right):
        """
        Overloaded ``!=`` operator

        :rtype: bool

        ``q1 != q2`` is True if ``q` is elementwise not equal to ``q2``.

        Examples::

            >>> q1 = Quaternion([1,2,3,4])
            >>> q2 = Quaternion([5,6,7,8])
            >>> q1 != q1
            False
            >>> q1 != q2
            True
            >>> Quaternion([np.r_[1,2,3,4], np.r_[5,6,7,8]]) != q1
            [False, True]
            >>> Quaternion([np.r_[1,2,3,4], np.r_[5,6,7,8]]) != q2
            [True, False]

        :seealso: :func:`__ne__`, :func:`spatialmath.base.quaternions.isequal`
        """
        assert isinstance(left, type(right)), 'operands to == are of different types'
        return left.binop(right, lambda x, y: not quat.isequal(x, y), list1=False)

    def __mul__(left, right):
        """
        Overloaded ``*`` operator

        :arg left: left multiplicand
        :type left: Quaternion
        :arg right: right multiplicand
        :type left: Quaternion, UnitQuaternion, float
        :return: product
        :rtype: Quaternion
        :raises: ValueError

        - ``q1 * q2`` is the Hamilton product of two quaternions
        - ``q * s`` is the scalar product, where ``s`` is a scalar

        ==============   ==============   ==============  ================
                   Multiplicands                   Product
        -------------------------------   --------------------------------
            left             right            type           result
        ==============   ==============   ==============  ================
        Quaternion       Quaternion       Quaternion      Hamilton product
        Quaternion       UnitQuaternion   Quaternion      Hamilton product
        Quaternion       scalar           Quaternion      scalar product
        ==============   ==============   ==============  ================

        Any other input combinations result in a ValueError.

        Note that left and right can have a length greater than 1 in which case:

        ====   =====   ====  ================================
        left   right   len     operation
        ====   =====   ====  ================================
         1      1       1    ``prod = left * right``
         1      N       N    ``prod[i] = left * right[i]``
         N      1       N    ``prod[i] = left[i] * right``
         N      N       N    ``prod[i] = left[i] * right[i]``
         N      M       -    ``ValueError``
        ====   =====   ====  ================================

        Examples::

            >>> Quaternion([1,2,3,4]) * Quaternion([5,6,7,8])
            -60.000000 < 12.000000, 30.000000, 24.000000 >

            >>> Quaternion([1,2,3,4]) * 2
            2.000000 < 4.000000, 6.000000, 8.000000 >
            >>> Quaternion([np.r_[1,2,3,4], np.r_[5,6,7,8]]) * 2
            2.000000 < 4.000000, 6.000000, 8.000000 >
            10.000000 < 12.000000, 14.000000, 16.000000 >

            >>> Quaternion([np.r_[1,2,3,4], np.r_[5,6,7,8]]) * Quaternion([1,2,3,4])
            -28.000000 < 4.000000, 6.000000, 8.000000 >
            -60.000000 < 20.000000, 14.000000, 32.000000 >
            >>> Quaternion([1,2,3,4]) * Quaternion([np.r_[1,2,3,4], np.r_[5,6,7,8]])
            -28.000000 < 4.000000, 6.000000, 8.000000 >
            -60.000000 < 12.000000, 30.000000, 24.000000 >
            >>> Quaternion([np.r_[1,2,3,4], np.r_[5,6,7,8]]) * Quaternion([np.r_[1,2,3,4], np.r_[5,6,7,8]])
            -28.000000 < 4.000000, 6.000000, 8.000000 >
            -124.000000 < 60.000000, 70.000000, 80.000000 >

        :seealso: :func:`__rmul__`, :func:`__imul__`, :func:`spatialmath.base.qqmul`
        """
        if isinstance(right, left.__class__):
            # quaternion * [unit]quaternion case
            return Quaternion(left.binop(right, quat.qqmul))

        elif argcheck.isscalar(right):
            # quaternion * scalar case
            #print('scalar * quat')
            return Quaternion([right * q._A for q in left])

        else:
            raise ValueError('operands to * are of different types')

    def __rmul__(right, left):
        """
        Overloaded ``*`` operator

        :arg right: right multiplicand
        :type right: Quaternion,
        :arg left: left multiplicand
        :type left: float
        :return: product
        :rtype: Quaternion
        :raises: ValueError

        ``s * q`` is the scalar product, where ``s`` is a scalar.

        Examples::

            >>> 2 * Quaternion([1,2,3,4])
            2.000000 < 4.000000, 6.000000, 8.000000 >
            >>> 2 * Quaternion([np.r_[1,2,3,4], np.r_[5,6,7,8]])
            2.000000 < 4.000000, 6.000000, 8.000000 >
            10.000000 < 12.000000, 14.000000, 16.000000 >

        :seealso: :func:`__mul__`
        """
        # scalar * quaternion case
        return Quaternion([left * q._A for q in right])

    def __imul__(left, right):
        """
        Overloaded ``*=`` operator

        :arg left: left multiplicand
        :type left: Quaternion
        :arg right: right multiplicand
        :type right: Quaternion, UnitQuaternion, float
        :return: product
        :rtype: Quaternion
        :raises: ValueError

        ``q1 *= q2`` sets ``q1 := q1 * q2``
        ``q1 *= s`` sets ``q1 := q1 * s`` where ``s`` is a scalar

        Example::

            >>> q = Quaternion([1,2,3,4])
            >>> q *= Quaternion([5,6,7,8])
            >>> q
            -60.000000 < 12.000000, 30.000000, 24.000000 >

            >>> q *= 2
            >>> q
            -120.000000 < 24.000000, 60.000000, 48.000000 >

        :seealso: :func:`__mul__`
        """
        return left.__mul__(right)

    def __pow__(self, n):
        """
        Overloaded ``**`` operator

        :rtype: Quaternion instance

        ``q ** N`` computes the product of ``q`` with itself ``N-1`` times, where ``N`` must be
        an integer.  If ``N``<0 the result is conjugated.

        Examples::

            >>> Quaternion([1,2,3,4]) ** 2
            -28.000000 < 4.000000, 6.000000, 8.000000 >
            >>> Quaternion([1,2,3,4]) ** -1
            1.000000 < -2.000000, -3.000000, -4.000000 >
            >>> Quaternion([np.r_[1,2,3,4], np.r_[5,6,7,8]]) ** 2
            -28.000000 < 4.000000, 6.000000, 8.000000 >
            -124.000000 < 60.000000, 70.000000, 80.000000 >

        :seealso: :func:`spatialmath.base.pow`
        """
        return self.__class__([quat.qpow(q._A, n) for q in self])

    def __ipow__(self, n):
        """
        Overloaded ``=**`` operator

        :rtype: Quaternion instance

        ``q **= N`` computes the product of ``q`` with itself ``N-1`` times, where ``N`` must be
        an integer.  If ``N``<0 the result is conjugated.

        Examples::

            >>> q = Quaternion([1,2,3,4])
            >>> q **= 2
            >>> q
            -28.000000 < 4.000000, 6.000000, 8.000000 >

            >>> q = Quaternion([np.r_[1,2,3,4], np.r_[5,6,7,8]])
            >>> q **= 2
            >>> q
            -28.000000 < 4.000000, 6.000000, 8.000000 >
            -124.000000 < 60.000000, 70.000000, 80.000000 >

        :seealso: :func:`__pow__`
        """

        return self.__pow__(n)

    def __truediv__(self, other):
        return NotImplemented  # Quaternion division not supported

    def __add__(left, right):
        """
        Overloaded ``+`` operator

        :arg left: left addend
        :type left: Quaternion, UnitQuaternion
        :arg right: right addend
        :type right: Quaternion, UnitQuaternion, float
        :return: sum
        :rtype: Quaternion, UnitQuaternion
        :raises: ValueError

        ==============   ==============   ==============  ===================
                   Operands                            Sum
        -------------------------------   -----------------------------------
            left             right            type           result
        ==============   ==============   ==============  ===================
        Quaternion       Quaternion       Quaternion      elementwise sum
        Quaternion       UnitQuaternion   Quaternion      elementwise sum
        Quaternion       scalar           Quaternion      add to each element
        UnitQuaternion   Quaternion       Quaternion      elementwise sum
        UnitQuaternion   UnitQuaternion   Quaternion      elementwise sum
        UnitQuaternion   scalar           Quaternion      add to each element
        ==============   ==============   ==============  ===================

        Any other input combinations result in a ValueError.

        Note that left and right can have a length greater than 1 in which case:

        ====   =====   ====  ================================
        left   right   len     operation
        ====   =====   ====  ================================
         1      1       1    ``prod = left + right``
         1      N       N    ``prod[i] = left + right[i]``
         N      1       N    ``prod[i] = left[i] + right``
         N      N       N    ``prod[i] = left[i] + right[i]``
         N      M       -    ``ValueError``
        ====   =====   ====  ================================

        A scalar of length N is a list, tuple or numpy array.
        A 3-vector of length N is a 3xN numpy array, where each column is a 3-vector.

        Examples::

            >>> Quaternion([1,2,3,4]) + Quaternion([5,6,7,8])
            6.000000 < 8.000000, 10.000000, 12.000000 >
            >>> Quaternion([np.r_[1,2,3,4], np.r_[5,6,7,8]]) + Quaternion([1,2,3,4])
            2.000000 < 4.000000, 6.000000, 8.000000 >
            6.000000 < 8.000000, 10.000000, 12.000000 >
            >>> Quaternion([np.r_[1,2,3,4], np.r_[5,6,7,8]]) + Quaternion([np.r_[1,2,3,4], np.r_[5,6,7,8]])
            2.000000 < 4.000000, 6.000000, 8.000000 >
            10.000000 < 12.000000, 14.000000, 16.000000 >
        """
        # results is not in the group, return an array, not a class
        assert isinstance(left, type(right)), 'operands to + are of different types'
        return Quaternion(left.binop(right, lambda x, y: x + y))

    def __sub__(left, right):
        """
        Overloaded ``-`` operator

        :arg left: left minuend
        :type left: Quaternion, UnitQuaternion
        :arg right: right subtahend
        :type right: Quaternion, UnitQuaternion, float
        :return: difference
        :rtype: Quaternion, UnitQuaternion
        :raises: ValueError

        ==============   ==============   ==============  ==========================
                   Operands                          Difference
        -------------------------------   ------------------------------------------
            left             right            type           result
        ==============   ==============   ==============  ==========================
        Quaternion       Quaternion       Quaternion      elementwise sum
        Quaternion       UnitQuaternion   Quaternion      elementwise sum
        Quaternion       scalar           Quaternion      subtract from each element
        UnitQuaternion   Quaternion       Quaternion      elementwise sum
        UnitQuaternion   UnitQuaternion   Quaternion      elementwise sum
        UnitQuaternion   scalar           Quaternion      subtract from each element
        ==============   ==============   ==============  ==========================

        Any other input combinations result in a ValueError.

        Note that left and right can have a length greater than 1 in which case:

        ====   =====   ====  ================================
        left   right   len     operation
        ====   =====   ====  ================================
         1      1       1    ``prod = left - right``
         1      N       N    ``prod[i] = left - right[i]``
         N      1       N    ``prod[i] = left[i] - right``
         N      N       N    ``prod[i] = left[i] - right[i]``
         N      M       -    ``ValueError``
        ====   =====   ====  ================================

        A scalar of length N is a list, tuple or numpy array.
        A 3-vector of length N is a 3xN numpy array, where each column is a 3-vector.

        Examples::

            >>> Quaternion([1,2,3,4]) - Quaternion([5,6,7,8])
            -4.000000 < -4.000000, -4.000000, -4.000000 >

            >>> Quaternion([np.r_[1,2,3,4], np.r_[5,6,7,8]]) - Quaternion([1,2,3,4])
            0.000000 < 0.000000, 0.000000, 0.000000 >
            4.000000 < 4.000000, 4.000000, 4.000000 >

            >>> Quaternion([np.r_[1,2,3,4], np.r_[5,6,7,8]]) - Quaternion([np.r_[1,2,3,4], np.r_[5,6,7,8]])
            0.000000 < 0.000000, 0.000000, 0.000000 >
            0.000000 < 0.000000, 0.000000, 0.000000 >

        """
        # results is not in the group, return an array, not a class
        # TODO allow class +/- a conformant array
        assert isinstance(left, type(right)), 'operands to - are of different types'
        return Quaternion(left.binop(right, lambda x, y: x - y))

    def __neg__(self):
        r"""
        Overloaded unary ``-`` operator

        :rtype: Quaternion or UnitQuaternion

        ``-q`` is a quaternion with all its components negated.

        Examples::

            >>> -Quaternion([1,2,3,4])
            -0.182574 << -0.365148, -0.547723, -0.730297 >>

            >>> -Quaternion([np.r_[1,2,3,4], np.r_[5,6,7,8]])
            -0.182574 << -0.365148, -0.547723, -0.730297 >>
            -0.379049 << -0.454859, -0.530669, -0.606478 >>
        """

        return UnitQuaternion([-x for x in self.data])  # pylint: disable=invalid-unary-operand-type

    def __repr__(self):
        """
        Readable representation of pose (superclass method)

        :return: readable representation of the pose as a list of arrays
        :rtype: str

        Example::

            >>> q = Quaternion([1,2,3,4])
            >>> q
            Quaternion(array([[ 1.        ,  0.        ,  0.        ,  0.        ],
                       [ 0.        ,  0.95533649, -0.29552021,  0.        ],
                       [ 0.        ,  0.29552021,  0.95533649,  0.        ],
                       [ 0.        ,  0.        ,  0.        ,  1.        ]]))

        """
        name = type(self).__name__
        if len(self) == 0:
            return name + '([])'
        elif len(self) == 1:
            # need to indent subsequent lines of the native repr string by 4 spaces
            return name + '(' + self._A.__repr__() + ')'
        else:
            # format this as a list of ndarrays
            return name + '([\n  ' + ',\n  '.join([v.__repr__() for v in self.data]) + ' ])'

    def _repr_pretty_(self, p, cycle):
        """
        Pretty string for IPython (superclass method)

        :param p: pretty printer handle (ignored)
        :param cycle: pretty printer flag (ignored)

        Print colorized output when variable is displayed in IPython, ie. on a line by
        itself.

        Example::

            In [1]: x

        """
        print(self.__str__())

    def __str__(self):
        """
        Pretty string representation of quaternion

        :return: readable representation of quaternion
        :rtype: str

        Format the quaternion elements into a single line format.  For example::

            >>> q = Quaternion([1,2,3,4])
            >>> print(x)
            1.000000 < 2.000000, 3.000000, 4.000000 >
            >> q = UnitQuaternion.Rx(0.3)
            0.988771 << 0.149438, 0.000000, 0.000000 >>

            Note that unit quaternions are denoted by different delimiters for
            the vector part.
        """
        if isinstance(self, UnitQuaternion):
            delim = ('<<', '>>')
        else:
            delim = ('<', '>')
        return '\n'.join([quat.qprint(q, file=None, delim=delim) for q in self.data])


class UnitQuaternion(Quaternion):
    r"""

    A unit quaternion has 2 parts, a scalar :math:`s`, and a 3-vector :math:`v` and is typically written as

    :math:`q = s \langle v_x, v_y, v_z \rangle`

    and has a unit length constraint, that is, :math:`s^2+v_x^2+v_y^2+v_z^2 = 1`.

    A unit-quaternion can be considered as a rotation :math:`\theta` about a
    unit-vector in space :math:`v=[v_x, v_y, v_z]`, so the unit quaternion can also be
    written as :math:`q = \cos \theta/2 \sin \theta/2 <v_x v_y v_z>`.

    The quaternion :math:`q` and :math:`-q` represent the equivalent rotation, and this is referred to
    as a double mapping.

 .. inheritance-diagram:: spatialmath.quaternion.UnitQuaternion
    :top-classes: collections.UserList
    :parts: 1

    The ``UnitQuaternion`` class inherits many methods from the ``Quaternion`` class

    """

    def __init__(self, s: Any = None, v=None, norm=True, check=True):
        """
        Construct a UnitQuaternion instance

        :arg norm: explicitly normalize the quaternion [default True]
        :type norm: bool
        :arg check: explicitly check dimension of passed lists [default True]
        :type check: bool
        :return: new unit uaternion
        :rtype: UnitQuaternion
        :raises: ValueError

        - ``UnitQuaternion()`` constructs the identity quaternion 1<0,0,0>
        - ``UnitQuaternion(s, v)`` constructs a unit quaternion with specified
          real ``s`` and ``v`` vector parts. ``v`` is a 3-vector given as a
          list, tuple, numpy.ndarray
        - ``UnitQuaternion(v)`` constructs a unit quaternion with specified
          elements from ``v`` which is a 4-vector given as a list, tuple, numpy.ndarray
        - ``UnitQuaternion(R)`` constructs a unit quaternion from an SO(2)
          rotation matrix given as a 3x3 numpy.ndarray. If ``check`` is True
          test the matrix for orthogonality.
        - ``UnitQuaternion(T)`` constructs a unit quaternion from an SE(3)
          homogeneous transformation matrix given as a 4x4 numpy.ndarray. If ``check`` is True
          test the matrix for orthogonality.
        - ``UnitQuaternion(X)`` constructs a unit quaternion from the rotational
          part of ``X`` which is an SO3 or SE3 instance.  If len(X) > 1 then
          the resulting unit quaternion is of the same length.
        - ``UnitQuaternion([q1, q2 .. qN])`` construct a new unit quaternion with ``N`` values where each element is a 4-vector
        - ``UnitQuaternion([Q1, Q2 .. QN])`` construct a new unit quaternion with ``N`` values where each element is a UnitQuaternion instance
        - ``UnitQuaternion([X1, X2 .. XN])`` construct a new unit quaternion with ``N`` values where each element is an SO3 or SE3 instance
        - ``UnitQuaternion(M)`` construct a new unit quaternion with ``N`` values where ``Q`` is a 4xN NumPy array.

        """
        super().__init__()

        if v is None:
            # single argument
            if super().arghandler(s, check=check):
                if norm:
                    self.data = [quat.unit(q) for q in self.data]
                return

            elif isinstance(s, np.ndarray) and tr.isrot(s, check=check):
                # UnitQuaternion(R) R is 3x3 rotation matrix
                self.data = [quat.r2q(s)]

            elif isinstance(s, np.ndarray) and tr.ishom(s, check=check):
                # UnitQuaternion(T) T is 4x4 homogeneous transformation matrix
                self.data = [quat.r2q(tr.t2r(s))]

            elif isinstance(s, np.ndarray) and s.shape[1] == 4:
                if norm:
                    self.data = [quat.qnorm(x) for x in s]
                else:
                    self.data = [x for x in s]

            elif isinstance(s, SO3):
                # UnitQuaternion(x) x is SO3 or SE3
                self.data = [quat.r2q(x.R) for x in s]

            elif isinstance(s[0], SO3):
                # list of SO3/SE3
                self.data = [quat.r2q(x.R) for x in s]

            else:
                raise ValueError('bad argument to UnitQuaternion constructor')

        elif argcheck.isscalar(s) and argcheck.isvector(v, 3):
            # UnitQuaternion(s, v)   s is scalar, v is 3-vector
            q = np.r_[s, argcheck.getvector(v)]
            if norm:
                q = quat.unit(q)
            self.data = [q]
        
        else:
            raise ValueError('bad argument to UnitQuaternion constructor')


    @staticmethod
    def _identity():
        return quat.eye()

    @staticmethod
    def isvalid(x, check=True):
        """
        Test if matrix is valid unit quaternion

        :param x: matrix to test
        :type x: numpy.ndarray
        :return: true of the matrix is 4x1.
        :rtype: bool
        """
        return x.shape == (4,) and (not check or tr.isunitvec(x))

    @property
    def R(self):
        """
        Unit quaternion as rotation matrix

        :return: equivalent rotational matrix
        :rtype: numpy.ndarray, shape=(3,3)

        ``q.R`` returns the rotation matrix which describes the equivalent rotation. If ``len(x)`` is:

            - 1, return an ndarray with shape=(3,3)
            - N>1, return ndarray with shape=(N,3,3)
        """
        return quat.q2r(self._A)

    @property
    def vec3(self):
        r"""
        Unit quaternion unique vector part

        :return: vector part of unit quaternion
        :rtype: numpy array, shape=(3,)

        ``q.vec3`` is the vector part of a unit quaternion.  If ``q`` has a negative scalar
        part we take the vector part of equivalent unit quaternion with a positive scalar part ``-q``.

        This vector part is a minimal unique representation of the unit quaternion and can be used in
        optimization procedures such as bundle adjustment.

        Examples::

            >>> q = UnitQuaternion.Rz(-4)
            >>> q
            -0.416147 << 0.000000, 0.000000, -0.909297 >>
            >>> q.vec3
            array([-0.        , -0.        ,  0.90929743])
            >>> q2 = UnitQuaternion.Vec3(q.vec3)
            >>> q2
            0.416147 << -0.000000, -0.000000, 0.909297 >>
            >>> q == q2
            True

        :seealso: :func:`~spatialmath.quaternion.UnitQuaternion.Vec3`
        """
        return quat.q2v(self._A)

    # -------------------------------------------- constructor variants
    @classmethod
    def Rx(cls, angle, unit='rad'):
        """
        Construct a UnitQuaternion object representing rotation about X-axis

        :arg angle: rotation angle
        :type angle: float or array_like
        :arg unit: rotation unit 'rad' [default] or 'deg'
        :type unit: str
        :return: new unit-quaternion
        :rtype: UnitQuaternion

        - ``UnitQuaternion(theta)`` constructs a unit quaternion representing a
          rotation of `theta` radians about the X-axis.
        - ``UnitQuaternion(theta, 'deg')`` constructs a unit quaternion representing a
          rotation of `theta` degrees about the X-axis.

        Examples::

            >>> UnitQuaternion.Rx(0.3)
            0.988771 << 0.149438, 0.000000, 0.000000 >>

            >>> UnitQuaternion.Rx([0, 0.3, 0.6])
            1.000000 << 0.000000, 0.000000, 0.000000 >>
            0.988771 << 0.149438, 0.000000, 0.000000 >>
            0.955336 << 0.295520, 0.000000, 0.000000 >>
        """
        angles = argcheck.getunit(argcheck.getvector(angle), unit)
        return cls([np.r_[math.cos(a / 2), math.sin(a / 2), 0, 0] for a in angles], check=False)

    @classmethod
    def Ry(cls, angle, unit='rad'):
        """
        Construct a UnitQuaternion object representing rotation about Y-axis

        :arg angle: rotation angle
        :type angle: float or array_like
        :arg unit: rotation unit 'rad' [default] or 'deg'
        :type unit: str
        :return: new unit-quaternion
        :rtype: UnitQuaternion

        - ``UnitQuaternion(theta)`` constructs a unit quaternion representing a
          rotation of `theta` radians about the Y-axis.
        - ``UnitQuaternion(theta, 'deg')`` constructs a unit quaternion representing a
          rotation of `theta` degrees about the Y-axis.

        Examples::

            >>> UnitQuaternion.Ry(0.3)
            0.988771 << 0.000000, 0.149438, 0.000000 >>

            >>> UnitQuaternion.Ry([0, 0.3, 0.6])
            1.000000 << 0.000000, 0.000000, 0.000000 >>
            0.988771 << 0.000000, 0.149438, 0.000000 >>
            0.955336 << 0.000000, 0.295520, 0.000000 >>
        """
        angles = argcheck.getunit(argcheck.getvector(angle), unit)
        return cls([np.r_[math.cos(a / 2), 0, math.sin(a / 2), 0] for a in angles], check=False)

    @classmethod
    def Rz(cls, angle, unit='rad'):
        """
        Construct a UnitQuaternion object representing rotation about Z-axis

        :arg angle: rotation angle
        :type angle: float or array_like
        :arg unit: rotation unit 'rad' [default] or 'deg'
        :type unit: str
        :return: new unit-quaternion
        :rtype: UnitQuaternion

        - ``UnitQuaternion(theta)`` constructs a unit quaternion representing a
          rotation of `theta` radians about the Z-axis.
        - ``UnitQuaternion(theta, 'deg')`` constructs a unit quaternion representing a
          rotation of `theta` degrees about the Z-axis.

        Examples::

            >>> UnitQuaternion.Rz(0.3)
            0.988771 << 0.000000, 0.000000, 0.149438 >>

            >>> UnitQuaternion.Rz([0, 0.3, 0.6])
            1.000000 << 0.000000, 0.000000, 0.000000 >>
            0.988771 << 0.000000, 0.000000, 0.149438 >>
            0.955336 << 0.000000, 0.000000, 0.295520 >>
        """
        angles = argcheck.getunit(argcheck.getvector(angle), unit)
        return cls([np.r_[math.cos(a / 2), 0, 0, math.sin(a / 2)] for a in angles], check=False)

    @classmethod
    def Rand(cls, N=1):
        """
        Construct a new random unit quaternion

        :param N: number of random rotations
        :type N: int
        :return: new unit-quaternion
        :rtype: UnitQuaternion

        - ``UnitQuaternion.Rand()`` is a uniformly distributed random unit quaternion value.
        - ``SO3.Rand(N)`` is a unit quaternion instance containing a sequence of N random unit quaternion
          values.

        Examples::

            >>> UnitQuaternion.Rand()
            0.622093 << -0.679361, 0.337190, -0.194349 >>

            >>> UnitQuaternion.Rand(3)
            0.117153 << -0.838230, 0.219071, -0.485442 >>
            -0.088206 << -0.397185, 0.852524, -0.328127 >>
            -0.204108 << -0.203155, -0.687019, 0.667138 >>

        :seealso: :func:`spatialmath.quaternion.UnitQuaternion.Rand`
        """
        return cls([quat.rand() for i in range(0, N)], check=False)

    @classmethod
    def Eul(cls, angles, *, unit='rad'):
        r"""
        Construct a new unit quaternion from Euler angles

        :param angles: 3-vector of Euler angles
        :type angles: array_like
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :return: new unit-quaternion
        :rtype: UnitQuaternion

        ``UnitQuaternion.Eul(ANGLES)`` is a unit quaternion that describes the 3D rotation defined by a 3-vector of Euler angles :math:`(\phi, \theta, \psi)` which
        correspond to consecutive rotations about the Z, Y, Z axes respectively.

        Examples::

            >>> UnitQuaternion.Eul([0.1, 0.2, 0.3])
            0.975170 << 0.009967, 0.099335, 0.197677 >>

        :seealso: :func:`~spatialmath.quaternion.UnitQuaternion.RPY`, :func:`~spatialmath.pose3d.SE3.eul`, :func:`~spatialmath.pose3d.SE3.Eul`, :func:`spatialmath.base.transforms3d.eul2r`
        """
        return cls(quat.r2q(tr.eul2r(angles, unit=unit)), check=False)

    @classmethod
    def RPY(cls, angles, *, order='zyx', unit='rad'):
        """
        Construct a new unit quaternion from roll-pitch-yaw angles

        :param angles: 3-vector of roll-pitch-yaw angles
        :type angles: array_like
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :param unit: rotation order: 'zyx' [default], 'xyz', or 'yxz'
        :type unit: str
        :return: new unit-quaternion
        :rtype: UnitQuaternion

        ``UnitQuaternion.RPY(ANGLES)`` is a unit quaternion that describes the 3D rotation defined by a  3-vector of roll, pitch, yaw angles :math:`(r, p, y)`
        which correspond to successive rotations about the axes specified by ``order``:

            - 'zyx' [default], rotate by yaw about the z-axis, then by pitch about the new y-axis,
              then by roll about the new x-axis.  Convention for a mobile robot with x-axis forward
              and y-axis sideways.
            - 'xyz', rotate by yaw about the x-axis, then by pitch about the new y-axis,
              then by roll about the new z-axis. Covention for a robot gripper with z-axis forward
              and y-axis between the gripper fingers.
            - 'yxz', rotate by yaw about the y-axis, then by pitch about the new x-axis,
              then by roll about the new z-axis. Convention for a camera with z-axis parallel
              to the optic axis and x-axis parallel to the pixel rows.

        Examples::

            >>> UnitQuaternion.RPY([0.1, 0.2, 0.3])
            0.983347 << 0.034271, 0.106021, 0.143572 >>

        :seealso: :func:`~spatialmath.quaternion.UnitQuaternion.Eul`, :func:`~spatialmath.pose3d.SE3.rpy`, :func:`~spatialmath.pose3d.SE3.RPY`, :func:`spatialmath.base.transforms3d.rpy2r`
        """
        return cls(quat.r2q(tr.rpy2r(angles, unit=unit, order=order)), check=False)

    @classmethod
    def OA(cls, o, a):
        """
        Construct a new unit quaternion from two vectors

        :param o: 3-vector parallel to Y- axis
        :type o: array_like
        :param a: 3-vector parallel to the Z-axis
        :type a: array_like
        :return: new unit-quaternion
        :rtype: UnitQuaternion

        ``SO3.OA(O, A)`` is a unit quaternion that describes the 3D rotation defined in terms of
        vectors parallel to the Y- and Z-axes of its reference frame.  In robotics these axes are
        respectively called the orientation and approach vectors defined such that
        R = [N O A] and N = O x A.

        Notes:

        - The A vector is the only guaranteed to have the same direction in the resulting
          rotation matrix
        - O and A do not have to be unit-length, they are normalized
        - O and A do not have to be orthogonal, so long as they are not parallel
        - The vectors O and A are parallel to the Y- and Z-axes of the equivalent coordinate frame.

        Examples::

            >>> UnitQuaternion.OA([0,0,-1], [0,1,0])
            0.707107 << -0.707107, 0.000000, -0.000000 >>

        :seealso: :func:`spatialmath.base.transforms3d.oa2r`
        """
        return cls(quat.r2q(tr.oa2r(o, a)), check=False)

    @classmethod
    def AngVec(cls, theta, v, *, unit='rad'):
        r"""
        Construct a new unit quaternion from rotation angle and axis

        :param theta: rotation
        :type theta: float
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :param v: rotation axis, 3-vector
        :type v: array_like
        :return: new unit-quaternion
        :rtype: UnitQuaternion

        ``SO3.AngVec(θ, v)`` is a unit quaternion that describes the 3D rotation
        defined by a rotation of ``θ`` about the 3-vector ``v``.

        Notes:

        - If :math:`\theta = 0` then return an identity unit quaternion,
        - Otherwise :math:`\lVert v \rVert > 0`.
        - :math:`v` does not have to be a unit vector.

        Examples::

            >>> UnitQuaternion.AngVec(0, [1,0,0])
            1.000000 << 0.000000, 0.000000, 0.000000 >>
            >>> UnitQuaternion.AngVec(90, [1,0,0], unit='deg')
            0.707107 << 0.707107, 0.000000, 0.000000 >>

        :seealso: :func:`~spatialmath.pose3d.SE3.angvec`, 
        :func:`spatialmath.base.transforms3d.angvec2r`
        """
        v = argcheck.getvector(v, 3)
        argcheck.isscalar(theta)
        theta = argcheck.getunit(theta, unit)
        return cls(s=math.cos(theta / 2), v=math.sin(theta / 2) * v, norm=False, check=False)

    @classmethod
    def EulerVec(cls, w):
        r"""
        Construct a new unit quaternion from Euler rotation vector

        :param w: rotation axis
        :type w: 3-element array_like
        :return: new unit-quaternion
        :rtype: UnitQuaternion

        ``SO3.EulerVec(ω)`` is a unit quaternion that describes the 3D rotation
        defined by a rotation of :math:`\theta = \lVert \omega \rVert` about the
        unit 3-vector :math:`\omega / \lVert \omega \rVert`.

        Notes:

        - If :math:`\lVert \omega \rVert = 0` then return an identity unit quaternion,
        - Otherwise :math:`\lVert v \rVert > 0`.

        Examples::

            >>> UnitQuaternion.AngVec(0, [1,0,0])
            1.000000 << 0.000000, 0.000000, 0.000000 >>
            >>> UnitQuaternion.AngVec(90, [1,0,0], unit='deg')
            0.707107 << 0.707107, 0.000000, 0.000000 >>

        :seealso: :func:`~spatialmath.pose3d.SE3.angvec`, \
        :func:`spatialmath.base.transforms3d.angvec2r`
        """
        assert argcheck.isvector(w, 3), 'w must be a 3-vector'
        w = argcheck.getvector(w)
        theta = tr.norm(w)
        s = math.cos(theta / 2)
        v = math.sin(theta / 2) * tr.unitvec(w)
        return cls(s=s, v=v, check=False)

    @classmethod
    def Vec3(cls, vec):
        r"""
        Construct a new unit quaternion from its vector part

        :param vec: vector part of unit quaternion
        :type vec: 3-element array_like

        ``UnitQuaternion.Vec(v)`` is a new unit quaternion with the specified vector part
        and the scalar part is :math:`s = \sqrt{1 - v_x^2 - v_y^2 - v_z^2}`.  The unit quaternion
        will always have a positive scalar part.

        Examples::

            >>> q = UnitQuaternion.Rz(-4)
            >>> q
            -0.416147 << 0.000000, 0.000000, -0.909297 >>
            >>> q.vec3
            array([-0.        , -0.        ,  0.90929743])
            >>> q2 = UnitQuaternion.Vec3(q.vec3)
            >>> q2
            0.416147 << -0.000000, -0.000000, 0.909297 >>
            >>> q == q2
            True

        :seealso: :func:`~spatialmath.quaternion.UnitQuaternion.vec3`
        """
        return cls(quat.v2q(vec))

    def inv(self):
        """
        Inverse of unit quaternion

        :return: unit-quaternion
        :rtype: UnitQuaternion

        - ``q.inv()`` is the inverse of the unit-quaternion.  This is a group operation
          and the product of the unit-quaternion and its inverse is the identity quaternion.

        Examples::

            >>> UnitQuaternion.Rx(0.3).inv()
            0.988771 << -0.149438, -0.000000, -0.000000 >>
            >>> UnitQuaternion.Rx(0.3).inv() * UnitQuaternion.Rx(0.3)
            1.000000 << 0.000000, 0.000000, 0.000000 >>
            >>> UnitQuaternion.Rx([0.3, 0.6]).inv()
            0.988771 << -0.149438, -0.000000, -0.000000 >>
            0.955336 << -0.295520, -0.000000, -0.000000 >>

        """
        return UnitQuaternion([quat.conj(q._A) for q in self])

    @staticmethod
    def qvmul(qv1, qv2):
        """
        Multiply unit quaternions defined by unique vector parts

        :param qv1: vector representation of first multiplicand
        :type qv1: numpy array, shape=(3,)
        :param qv1: vector representation of second multiplicand
        :type qv1: numpy array, shape=(3,)

        ``UnitQuaternion(qv1, qv2)`` is the Hamilton product of two unit quaternions
        represented in minimal vector form.

        Examples::

            >>> q1 = UnitQuaternion.Rx(0.3)
            >>> q2 = UnitQuaternion.Ry(-0.3)
            >>> qv1 = q1.vec3
            >>> qv1
            array([0.14943813, 0.        , 0.        ])
            >>> qv2 = q2.vec3
            >>> qv = UnitQuaternion.qvmul(qv1, qv2)
            >>> qv
            array([ 0.1477601 , -0.1477601 , -0.02233176])
            >>> UnitQuaternion.Vec3(qv)
            0.977668 << 0.147760, -0.147760, -0.022332 >>
            >>> UnitQuaternion.Rx(0.3) * UnitQuaternion.Ry(-0.3)
            0.977668 << 0.147760, -0.147760, -0.022332 >>

        :seealso: :func:`~spatialmath.quaternion.UnitQuaternion.vec3`, :func:`~spatialmath.quaternion.UnitQuaternion.Vec3`
        """
        return quat.vvmul(qv1, qv2)

    def dot(self, omega):
        """
        Rate of change of unit quaternion

        :param omega: angular velocity in world frame
        :type omega: 3-element array_like
        :return: rate of change of unit quaternion
        :rtype: numpy.ndarray, shape=(4,)

        ``q.dot(ω)`` is the rate of change of the elements of the unit quaternion ``q``
        which represents the orientation of a body frame with angular velocity ``ω`` in
        the world frame.
        """
        return tr.dot(self._A, omega)

    def dotb(self, omega):
        """
        Rate of change of unit quaternion in body frame

        :param omega: angular velocity in body frame
        :type omega: 3-element array_like
        :return: rate of change of unit quaternion
        :rtype: numpy.ndarray, shape=(4,)

        ``q.dotb(ω)`` is the rate of change of the elements of the unit quaternion ``q``
        which represents the orientation of a body frame with angular velocity ``ω`` in
        the body frame.
        """
        return tr.dotb(self._A, omega)

    def __mul__(left, right):
        """
        Multiply unit quaternion

        :arg left: left multiplicand
        :type left: UnitQuaternion
        :arg right: right multiplicand
        :type left: UnitQuaternion, Quaternion, 3-vector, 3xN array, float
        :return: product
        :rtype: Quaternion, UnitQuaternion
        :raises: ValueError

        ==============   ==============   ==============  ================
                   Multiplicands                   Product
        -------------------------------   --------------------------------
            left             right            type           result
        ==============   ==============   ==============  ================
        UnitQuaternion   Quaternion       Quaternion      Hamilton product
        UnitQuaternion   UnitQuaternion   UnitQuaternion  Hamilton product
        UnitQuaternion   scalar           Quaternion      scalar product
        UnitQuaternion   3-vector         3-vector        vector rotation
        UnitQuaternion   3xN array        3xN array       vector rotations
        ==============   ==============   ==============  ================

        Any other input combinations result in a ValueError.

        Note that left and right can have a length greater than 1 in which case:

        ====   =====   ====  ================================
        left   right   len     operation
        ====   =====   ====  ================================
         1      1       1    ``prod = left * right``
         1      N       N    ``prod[i] = left * right[i]``
         N      1       N    ``prod[i] = left[i] * right``
         N      N       N    ``prod[i] = left[i] * right[i]``
         N      M       -    ``ValueError``
        ====   =====   ====  ================================

        A scalar of length N is a list, tuple or numpy array.
        A 3-vector of length N is a 3xN numpy array, where each column is 
        a 3-vector.

        Examples::

            >>> UnitQuaternion.Rx(0.3) * UnitQuaternion.Rx(0.3)
            0.955336 << 0.295520, 0.000000, 0.000000 >>
            >>> UnitQuaternion.Rx(0.3) * UnitQuaternion.Rx([0.3, 0.6])
            0.955336 << 0.295520, 0.000000, 0.000000 >>
            0.900447 << 0.434966, 0.000000, 0.000000 >>
            >>> UnitQuaternion.Rx([0.3, 0.6]) * UnitQuaternion.Rx(0.3)
            0.955336 << 0.295520, 0.000000, 0.000000 >>
            0.900447 << 0.434966, 0.000000, 0.000000 >>
            >>> UnitQuaternion.Rx([0.3, 0.6]) * UnitQuaternion.Rx([0.3, 0.6])
            0.955336 << 0.295520, 0.000000, 0.000000 >>
            0.825336 << 0.564642, 0.000000, 0.000000 >>

        :seealso: :func:`~spatialmath.Quaternion.__mul__`
        """
        if isinstance(left, right.__class__):
            # quaternion * quaternion case (same class)
            return right.__class__(left.binop(right, quat.qqmul))

        elif argcheck.isscalar(right):
            # quaternion * scalar case
            #print('scalar * quat')
            return Quaternion([right * q._A for q in left])

        elif isinstance(right, (list, tuple, np.ndarray)):
            # unit quaternion * vector
            #print('*: pose x array')
            if argcheck.isvector(right, 3):
                v = argcheck.getvector(right)
                if len(left) == 1:
                    # pose x vector
                    #print('*: pose x vector')
                    return quat.qvmul(left._A, argcheck.getvector(right, 3))

                elif len(left) > 1 and argcheck.isvector(right, 3):
                    # pose array x vector
                    #print('*: pose array x vector')
                    return np.array([tr.qvmul(x, v) for x in left._A]).T

            elif len(left) == 1 and isinstance(right, np.ndarray) and right.shape[0] == 3:
                # pose x stack of vectors
                return np.array([tr.qvmul(left._A, x) for x in right.T]).T
            else:
                raise ValueError('bad operands')
        else:
            raise ValueError('UnitQuaternion: operands to * are of different types')

    def __imul__(left, right):
        """
        Multiply unit quaternion in place

        :arg left: left multiplicand
        :type left: UnitQuaternion
        :arg right: right multiplicand
        :type right: UnitQuaternion, Quaternion, float
        :return: product
        :rtype: UnitQuaternion, Quaternion
        :raises: ValueError

        Multiplies a quaternion in place. If the right operand is a list,
        the result will be a list.

        Example::

            >>> q = UnitQuaternion.Rx(0.3)
            >>> q *= UnitQuaternion.Rx(0.3)
            >>> q
            0.955336 << 0.295520, 0.000000, 0.000000 >>

        :seealso: :func:`__mul__`

        """
        return left.__mul__(right)

    def __truediv__(left, right):
        """
        Overloaded ``/`` operator

        :rtype: Quaternion or UnitQuaternion

        ``q1 / q2`` is equivalent to ``q1 * q1.inv()``.

        Examples::

            >>> UnitQuaternion.Rx(0.3) / UnitQuaternion.Rx(0.3)
            1.000000 << 0.000000, 0.000000, 0.000000 >>
            >>> UnitQuaternion.Rx([0.3, 0.6]) / UnitQuaternion.Rx(0.3)
            1.000000 << 0.000000, 0.000000, 0.000000 >>
            0.988771 << 0.149438, 0.000000, 0.000000 >>

            >>> UnitQuaternion.Rx(0.3) / UnitQuaternion.Rx([0.3, 0.6])
            1.000000 << 0.000000, 0.000000, 0.000000 >>
            0.988771 << -0.149438, 0.000000, 0.000000 >>

            >>> UnitQuaternion.Rx([0.3, 0.6]) / UnitQuaternion.Rx([0.3, 0.6])
            1.000000 << 0.000000, 0.000000, 0.000000 >>
            1.000000 << 0.000000, 0.000000, 0.000000 >>
        """

        assert isinstance(left, type(right)), 'operands to / are of different types'
        return UnitQuaternion(left.binop(right, lambda x, y: tr.qqmul(x, tr.conj(y))))

    def __eq__(left, right):
        """
        Overloaded ``==`` operator

        :rtype: bool

        ``q1 == q2`` is True if ``q1` is elementwise equal to ``q2`` and accounts for the
        double mapping.

        Examples::

            >>> q1 = UnitQuaternion.Rx(0.3)
            >>> q2 = UnitQuaternion.Ry(0.3)
            >>> q1 == q1
            True
            >>> q1 == (-q1)
            True
            >>> q1 == q2
            False
            >>> UnitQuaternion([q1, q2]) == q1
            [True, False]
            >>> UnitQuaternion([q1, q2]) == q2
            [False, True]
            >>> UnitQuaternion([q1, q2]) == UnitQuaternion([q1, q2])
            [True, True]

        :seealso: :func:`__ne__`, :func:`spatialmath.base.quaternions.isequal`
        """
        return left.binop(right, lambda x, y: quat.isequal(x, y, unitq=True), list1=False)

    def __ne__(left, right):
        """
        Overloaded ``!=`` operator

        :rtype: bool

        ``q1 != q2`` is True if ``q` is elementwise not equal to ``q2``.

        Examples::

            >>> q1 = UnitQuaternion.Rx(0.3)
            >>> q2 = UnitQuaternion.Ry(0.3)
            >>> q1 != q1
            True
            >>> q1 != (-q1)
            False
            >> q1 != q2
            True
            >>> UnitQuaternion([q1, q2]) == q1
            [False, True]
            >>> UnitQuaternion([q1, q2]) == q2
            [True, False]
            >>> UnitQuaternion([q1, q2]) == UnitQuaternion([q1, q2])
            [False, False]

        :seealso: :func:`__ne__`, :func:`spatialmath.base.quaternions.isequal`
        """
        return left.binop(right, lambda x, y: not quat.isequal(x, y, unitq=True), list1=False)

    def interp(self, s=0, dest=None, shortest=False):
        """
        Algorithm source: https://en.wikipedia.org/wiki/Slerp
        :param qr: UnitQuaternion
        :param shortest: Take the shortest path along the great circle
        :param s: interpolation in range [0,1]
        :type s: float
        :return: interpolated UnitQuaternion
        """
        # TODO vectorize

        if dest is not None:
            # 2 quaternion form
            assert isinstance(dest, UnitQuaternion)
            if s == 0:
                return self
            elif s == 1:
                return dest
            q1 = self.vec
            q2 = dest.vec
        else:
            # 1 quaternion form
            if s == 0:
                return UnitQuaternion()
            elif s == 1:
                return self

            q1 = quat.eye()
            q2 = self.vec

        assert 0 <= s <= 1, 's must be in interval [0,1]'

        dot = quat.inner(q1, q2)

        # If the dot product is negative, the quaternions
        # have opposite handed-ness and slerp won't take
        # the shorter path. Fix by reversing one quaternion.
        if shortest:
            if dot < 0:
                q1 = - q1
                dot = -dot

        dot = np.clip(dot, -1, 1)  # Clip within domain of acos()
        theta_0 = math.acos(dot)  # theta_0 = angle between input vectors
        theta = theta_0 * s  # theta = angle between v0 and result
        if theta_0 == 0:
            return UnitQuaternion(q1)

        s1 = float(math.cos(theta) - dot * math.sin(theta) / math.sin(theta_0))
        s2 = math.sin(theta) / math.sin(theta_0)
        out = (q1 * s1) + (q2 * s2)
        return UnitQuaternion(out)

    def plot(self, *args, **kwargs):
        """
        Plot unit quaternion as a coordinate frame

        :param `**kwargs`: plotting options

        - ``q.plot()`` displays the orientation ``q`` as a coordinate frame in 3D.
          There are many options, see the links below.

        Example::

            >>> q = UnitQuaternion.Rx(0.3)
            >>> q.plot(frame='A', color='green')

        :seealso: :func:`~spatialmath.base.transforms3d.trplot`
        """
        tr.trplot(tr.q2r(self._A), *args, **kwargs)

    def animate(self, *args, **kwargs):
        """
        Plot unit quaternion as an animated coordinate frame

        :param start: initial pose, defaults to null/identity
        :type start: UnitQuaternion
        :param `**kwargs`: plotting options

        - ``q.animate()`` displays the orientation ``q`` as a coordinate frame moving
          from the origin in either 3D.  There are 
          many options, see the links below.
        - ``q.animate(*args, start=q1)`` displays the orientation ``q`` as a coordinate
          frame moving from orientation ``q11``, in 3D.  There are 
          many options, see the links below.

        Example::

            >>> X = UnitQuaternion.Rx(0.3)
            >>> X.animate(frame='A', color='green')
            >>> X.animate(start=UnitQuaternion.Ry(0.2))

        :see :func:`~spatialmath.base.transforms3d.tranimate`, :func:`~spatialmath.base.transforms3d.trplot`
        """

    def rpy(self, unit='rad', order='zyx'):
        """
        Unit quaternion as roll-pitch-yaw angles

        :param order: angle sequence order, default to 'zyx'
        :type order: str
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :return: 3-vector of roll-pitch-yaw angles
        :rtype: numpy.ndarray, shape=(3,)

        ``q.rpy`` is the roll-pitch-yaw angle representation of the 3D rotation.  The angles are
        a 3-vector :math:`(r, p, y)` which correspond to successive rotations about the axes
        specified by ``order``:

            - 'zyx' [default], rotate by yaw about the z-axis, then by pitch about the new y-axis,
              then by roll about the new x-axis.  Convention for a mobile robot with x-axis forward
              and y-axis sideways.
            - 'xyz', rotate by yaw about the x-axis, then by pitch about the new y-axis,
              then by roll about the new z-axis. Convention for a robot gripper with z-axis forward
              and y-axis between the gripper fingers.
            - 'yxz', rotate by yaw about the y-axis, then by pitch about the new x-axis,
              then by roll about the new z-axis. Convention for a camera with z-axis parallel
              to the optic axis and x-axis parallel to the pixel rows.

        If ``len(x)`` is:

        - 1, return an ndarray with shape=(3,)
        - N>1, return ndarray with shape=(N,3)

        Examples::

            >>> UnitQuaternion.Rx(0.3).rpy()
            array([ 0.3, -0. ,  0. ])
            >>> UnitQuaternion.Rz([0.2, 0.3]).rpy()
            array([[ 0. , -0. ,  0.2],
                [ 0. , -0. ,  0.3]])

        :seealso: :func:`~spatialmath.pose3d.SE3.RPY`, ::func:`spatialmath.base.transforms3d.tr2rpy`
        """
        if len(self) == 1:
            return tr.tr2rpy(self.R, unit=unit, order=order)
        else:
            return np.array([tr.tr2rpy(q.R, unit=unit, order=order) for q in self])

    def eul(self, unit='rad'):
        r"""
        Unit quaternion as Euler angles

        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :return: 3-vector of Euler angles
        :rtype: numpy.ndarray, shape=(3,)

        ``q.eul`` is the Euler angle representation of the rotation.  Euler angles are
        a 3-vector :math:`(\phi, \theta, \psi)` which correspond to consecutive
        rotations about the Z, Y, Z axes respectively.

        If ``len(x)`` is:

        - 1, return an ndarray with shape=(3,)
        - N>1, return ndarray with shape=(N,3)

        - ndarray with shape=(3,), if len(R) == 1
        - ndarray with shape=(N,3), if len(R) = N > 1

        Examples::

            >>> UnitQuaternion.Rz(0.3).eul()
            array([0. , 0. , 0.3])
            >>> UnitQuaternion.Ry([0.3, 0.4]).eul()
            array([[0. , 0.3, 0. ],
                [0. , 0.4, 0. ]])

        :seealso: :func:`~spatialmath.pose3d.SE3.Eul`, ::func:`spatialmath.base.transforms3d.tr2eul`
        """
        if len(self) == 1:
            return tr.tr2eul(self.R, unit=unit)
        else:
            return np.array([tr.tr2eul(q.R, unit=unit) for q in self])

    def angvec(self, unit='rad'):
        r"""
        Unit quaternion as angle and rotation vector

        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :param check: check that rotation matrix is valid
        :type check: bool
        :return: :math:`(\theta, {\bf v})`
        :rtype: float, numpy.ndarray, shape=(3,)

        ``q.angvec()`` is a tuple :math:`(\theta, v)` containing the rotation 
        angle and a rotation axis which is equivalent to the rotation of
        the unit quaternion ``q``.

        Example::

        >>> UnitQuaternion.Rz(0.3).angvec()
            (0.3, array([0., 0., 1.]))

        :seealso: :func:`~spatialmath.quaternion.AngVec`, :func:`~angvec2r`
        """
        return tr.tr2angvec(self.R, unit=unit)

    def SO3(self):
        """
        Unit quaternion as SO3 instance

        :return: an SO(3) representation
        :rtype: SO3 instance

        ``q.SO3()`` is an ``SO3`` instance representing the same rotation 
        as the unit quaternion ``q``.

        Examples::

            >>> UnitQuaternion.Rz(0.3).SO3()
            SO3(array([[ 0.95533649, -0.29552021,  0.        ],
                    [ 0.29552021,  0.95533649,  0.        ],
                    [ 0.        ,  0.        ,  1.        ]]))
        """
        return SO3(self.R, check=False)

    def SE3(self):
        """
        Unit quaternion as SE3 instance

        :return: an SE(3) representation
        :rtype: SE3 instance

        ``q.SE3()`` is an ``SE3`` instance representing the same rotation 
        as the unit quaternion ``q`` and with zero translation.

        Examples::

            >>> UnitQuaternion.Rz(0.3).SE3()
            SE3(array([[ 0.95533649, -0.29552021,  0.        ,  0.        ],
                    [ 0.29552021,  0.95533649,  0.        ,  0.        ],
                    [ 0.        ,  0.        ,  1.        ,  0.        ],
                    [ 0.        ,  0.        ,  0.        ,  1.        ]]))
        """
        return SE3(tr.r2t(self.R), check=False)


if __name__ == '__main__':  # pragma: no cover

    import pathlib

    exec(open(pathlib.Path(__file__).parent.absolute() / "test_quaternion.py").read())  # pylint: disable=exec-used
