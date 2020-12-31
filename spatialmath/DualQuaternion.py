import numpy as np
from spatialmath import Quaternion, UnitQuaternion, SE3
from spatialmath import base

# TODO scalar multiplication

class DualQuaternion:
    r"""
    A dual number is an ordered pair :math:`\hat{a} = (a, b)` or written as
    :math:`a + \epsilon b` where :math:`\epsilon^2 = 0`.

    A dual quaternion can be considered as either:

    - a quaternion with dual numbers as coefficients
    - a dual of quaternions, written as an ordered pair of quaternions

    The latter form is used here.

    :References:

    - http://web.cs.iastate.edu/~cs577/handouts/dual-quaternion.pdf
    - https://en.wikipedia.org/wiki/Dual_quaternion
    """

    def __init__(self, real=None, dual=None):
        """
        Construct a new dual quaternion

        :param real: real quaternion
        :type real: Quaternion or UnitQuaternion
        :param dual: dual quaternion
        :type dual: Quaternion or UnitQuaternion
        :raises ValueError: incorrect parameters

        Example:

        .. runblock:: pycon

            >>> from spatialmath import DualQuaternion, Quaternion
            >>> d = DualQuaternion(Quaternion([1,2,3,4]), Quaternion([5,6,7,8]))
            >>> print(d)
            >>> d = DualQuaternion([1, 2, 3, 4,  5, 6, 7, 8])
            >>> print(d)

        """

        if real is None and dual is None:
            self.real = None
            self.dual = None
            return
        elif dual is None and base.isvector(real, 8):
            self.real = Quaternion(real[0:4])
            self.dual = Quaternion(real[4:8])
        elif real is not None and dual is not None:
            if not isinstance(real, Quaternion):
                raise ValueError('real part must be a Quaternion subclass')
            if not isinstance(dual, Quaternion):
                raise ValueError('real part must be a Quaternion subclass')
            self.real = real  # quaternion, real part
            self.dual = dual  # quaternion, dual part
        else:
            raise ValueError('expecting zero or two parameters')

    def __repr__(self):
        return str(self)

    def __str__(self):
        """
        String representation of dual quaternion

        :return: compact string representation
        :rtype: str

        Example:

        .. runblock:: pycon

            >>> from spatialmath import DualQuaternion, Quaternion
            >>> d = DualQuaternion(Quaternion([1,2,3,4]), Quaternion([5,6,7,8]))
            >>> str(d)
        """
        return str(self.real) + " + ε " + str(self.dual)

    def norm(self):
        """
        Norm of a dual quaternion

        :return: Norm as a dual number
        :rtype: 2-tuple

        Example:

        .. runblock:: pycon

            >>> from spatialmath import DualQuaternion, Quaternion
            >>> d = DualQuaternion(Quaternion([1,2,3,4]), Quaternion([5,6,7,8]))
            >>> d.norm()  # norm is a dual number
        """
        a = self.real * self.real.conj()
        b= self.real * self.dual.conj() + self.dual * self.real.conj()
        return (a.s, b.s)

    def conj(self):
        r"""
        Conjugate of dual quaternion

        :return: Conjugate
        :rtype: DualQuaternion

        There are several conjugates defined for a dual quaternion. This one
        mirrors conjugation for a regular quaternion.  For the dual quaternion
        :math:`(p, q)` it returns :math:`(p^*, q^*)`.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import DualQuaternion, Quaternion
            >>> d = DualQuaternion(Quaternion([1,2,3,4]), Quaternion([5,6,7,8]))
            >>> d.conj()
        """
        return DualQuaternion(self.real.conj(), self.dual.conj())

    def __add__(left, right):  # lgtm[py/not-named-self] pylint: disable=no-self-argument
        """
        Sum of two dual quaternions

        :return: Product
        :rtype: DualQuaternion

        Example:

        .. runblock:: pycon

            >>> from spatialmath import DualQuaternion, Quaternion
            >>> d = DualQuaternion(Quaternion([1,2,3,4]), Quaternion([5,6,7,8]))
            >>> d + d
        """
        return DualQuaternion(left.real + right.real, left.dual + right.dual)

    def __sub__(left, right):  # lgtm[py/not-named-self] pylint: disable=no-self-argument
        """
        Difference of two dual quaternions

        :return: Product
        :rtype: DualQuaternion

        Example:

        .. runblock:: pycon

            >>> from spatialmath import DualQuaternion, Quaternion
            >>> d = DualQuaternion(Quaternion([1,2,3,4]), Quaternion([5,6,7,8]))
            >>> d - d
        """
        return DualQuaternion(left.real - right.real, left.dual - right.dual)

    def __mul__(left, right):  # lgtm[py/not-named-self] pylint: disable=no-self-argument
        """
        Product of two dual quaternions

        :return: Product
        :rtype: DualQuaternion

        Example:

        .. runblock:: pycon

            >>> from spatialmath import DualQuaternion, Quaternion
            >>> d = DualQuaternion(Quaternion([1,2,3,4]), Quaternion([5,6,7,8]))
            >>> d * d
        """
        real = left.real * right.real
        dual = left.real * right.dual + left.dual * right.real

        if isinstance(left, UnitDualQuaternion) and isinstance(left, UnitDualQuaternion):
            return UnitDualQuaternion(real, dual)
        else:
            return DualQuaternion(real, dual)

    def matrix(self):
        """
        Dual quaternion as a matrix

        :return: Matrix represensation
        :rtype: ndarray(8,8)

        Dual quaternion multiplication can also be written as a matrix-vector
        product.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import DualQuaternion, Quaternion
            >>> d = DualQuaternion(Quaternion([1,2,3,4]), Quaternion([5,6,7,8]))
            >>> d.matrix()
            >>> d.matrix() @ d.vec
            >>> d * d
        """
        return np.block([
                [self.real.matrix, np.zeros((4,4))],
                [self.dual.matrix, self.real.matrix]
            ])

    @property
    def vec(self):
        """
        Dual quaternion as a vector

        :return: Vector represensation
        :rtype: ndarray(8)

        Example:

        .. runblock:: pycon

            >>> from spatialmath import DualQuaternion, Quaternion
            >>> d = DualQuaternion(Quaternion([1,2,3,4]), Quaternion([5,6,7,8]))
            >>> d.vec
        """
        return np.r_[self.real.vec, self.dual.vec]


    # def log(self):
    #     pass
        
class UnitDualQuaternion(DualQuaternion):

    def __init__(self, real=None, dual=None):
        """
        Create new unit dual quaternion

        :param real: real quaternion or SE(3) matrix
        :type real: Quaternion, UnitQuaternion or SE3
        :param dual: dual quaternion
        :type dual: Quaternion or UnitQuaternion

        - ``UnitDualQuaternion(real, dual)`` is a new unit dual quaternion with
          real and dual parts as specified.
        - ``UnitDualQuaternion(T)`` is a new unit dual quaternion equivalent to
          the rigid-body motion described by the SE3 value ``T``.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import DualQuaternion
            >>> T = SE3.Rand()
            >>> print(T)
            >>> d = UnitDualQuaternion(T))
            >>> print(d)
            >>> type(d)
            >>> print(d.norm())  # norm is (1, 0)
        """
        if real is None and dual is None:
            self.real = None
            self.dual = None
            return
        elif real is not None and dual is not None:
            self.real = real  # quaternion, real part
            self.dual = dual  # quaternion, dual part
        elif dual is None and isinstance(real, SE3):
            T = real
            S = UnitQuaternion(T.R)
            D = Quaternion.Pure(T.t)
        
            self.real = S
            self.dual = 0.5 * D * S

    def T(self):
        """
        Convert unit dual quaternion to SE(3) matrix

        :return: SE(3) matrix
        :rtype: SE3

        Example:

        .. runblock:: pycon

            >>> from spatialmath import DualQuaternion
            >>> T = SE3.Rand()
            >>> print(T)
            >>> d = UnitDualQuaternion(T))
            >>> print(d)
            >>> print(d.T)
        """
        R = base.q2r(self.real.A)
        t = 2 * self.dual * self.real.conj()

        return SE3(base.rt2tr(R, t.v))
        
    # def exp(self):
    #     w = self.real.v
    #     v = self.dual.v
    #     theta = base.norm(w)

if __name__ == "__main__":

    a = DualQuaternion(Quaternion([1,2,3,4]), Quaternion([5,6,7,8]))
    print(a)
    print(a.vec)
    print(a.conj())
    print(a.norm())
    print(a+a)
    print(a*a)
    print(a.matrix())

    T = SE3.Rand()
    print(T)

    aa = UnitDualQuaternion(T)
    print(aa)
    print(aa.norm())
    print(aa.T())