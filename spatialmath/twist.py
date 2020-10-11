import numpy as np
import math

from spatialmath.pose3d import SO3, SE3
from spatialmath.pose2d import SO2, SE2
from spatialmath.geom3d import Plucker
import spatialmath.base as base
from spatialmath.smuserlist import SMUserList
from spatialmath.spatialvector import SpatialVelocity, SpatialAcceleration, SpatialForce

class SMTwist(SMUserList):
    """
    Superclass for 3D and 2D twist objects

    Subclasses are:

    - ``Twist3`` representing rigid-body motion in 3D as a 6-vector
    - ``Twist2`` representing rigid-body motion in 2D as a 3-vector

    A twist is the unique elements of the logarithm of the corresponding SE(N)
    matrix.

    Arithmetic operators are overloaded but the operation they perform depend
    on the types of the operands.  For example:

    - ``*`` will compose two instances of the same subclass, and the result will be
      an instance of the same subclass, since this is a group operator.

    These classes all inherit from ``UserList`` which enables them to 
    represent a sequence of values, ie. a ``Twist3`` instance can contain
    a sequence of twists.  Most of the Python ``list`` operators
    are applicable::

        >>> x = Twist3()  # new instance with zero value
        >>> len(x)     # it is a sequence of one value
        1
        >>> x.append(x)  # append to itself
        >>> len(x)       # it is a sequence of two values
        2
        >>> x[1]         # the element has a 4x4 matrix value
        Twist3([0, 0, 0, 0, 0, 0])
        >>> x[1] = SE3.Rx(0.3).Twist3()  # set an elements of the sequence
        >>> x.reverse()         # reverse the elements in the sequence
        >>> del x[1]            # delete an element

    References:

        - "Mechanics, planning and control"
          Park & Lynch, Cambridge, 2016.

    This class is subclassed for the 3D and 2D cases

    .. inheritance-diagram:: spatialmath.twist.Twist3 spatialmath.twist.Twist2
       :top-classes: collections.UserList
       :parts: 2

    """

    def __init__(self):
        super().__init__()   # enable UserList superpowers

    @property
    def S(self):
        """
        Twist as a vector (superclass property)

        :return: Twist vector
        :rtype: numpy.ndarray, shape=(N,)

        - ``X.S`` is a 3-vector if X is a ``Twist2`` instance, and a 6-vector if
          X is a ``Twist`` instance.

        .. notes::

            - the vector is the unique elements of the se(N) representation.
            - the vector is sometimes referred to as the twist coordinate vector.
            - if ``len(X)`` > 1 then return a list of vectors.
        """
        # get the underlying numpy array
        if len(self.data) == 1:
            return self.data[0]
        else:
            return self.data

    @property
    def isprismatic(self):
        """
        Test for prismatic twist (superclass property)

        :return: Whether twist is purely prismatic
        :rtype: bool

        Example::

            >>> x = Twist3.R([1,2,3], [4,5,6])
            >>> x.isprismatic
            False

        """
        if len(self) == 1:
            return base.iszerovec(self.w)
        else:
            return [base.iszerovec(x.w) for x in self.data]

    @property
    def unit(self):
        """
        Unitize twist (superclass property)

        :return: a unit twist
        :rtype: Twist3 or Twist2

        ``twist.unit()`` is a Twist object representing a unit aligned with the
        Twist ``twist``.
        """
        if base.iszerovec(self.w):
            # rotational twist
            return Twist3(self.S / base.norm(S.w))
        else:
            # prismatic twist
            return Twist3(base.unitvec(self.v), [0, 0, 0])

    @property
    def isunit(self):
        """
        Test for unit twist (superclass property)

        :return: If twist is a unit-twist
        :rtype: bool
        """
        if len(self) == 1:
            return base.isunittwist(self.S)
        else:
            return [base.isunittwist(x) for x in self.data]


# ======================================================================== #


class Twist3(SMTwist):
    """
    TWIST SE(2) and SE(3) Twist class

    A Twist class holds the parameters of a twist, a representation of a
    rigid body displacement in SE(2) or SE(3).

    """

    def __init__(self, arg=None, w=None, check=True):
        """
        Construct a new 3D twist object

        - ``Twist3()`` is a Twist3 instance representing null motion -- the
          identity twist
        - ``Twist3(t)`` is a Twist3 instance from an array-like (6,)
        - ``Twist3(v, w)`` is a Twist3 instance from a moment ``v`` (3,) and
           direction ``w`` (3,)
        - ``Twist3([t1, t2, ... tN])`` where each ti is a numpy array (6,)
        - ``Twist3([X1, X2, ... XN])`` where each Xi is a Twist3 instance, is a
          Twist3 instance containing N motions

        """
        super().__init__()

        if w is None:
            # zero or one arguments passed
            if super().arghandler(arg, convertfrom=(SE3,), check=check):
                return

        elif w is not None and base.isvector(w, 3) and base.isvector(arg,3):
            # Twist(v, w)
            self.data = [np.r_[arg, w]]
            return

        raise ValueError('bad twist value')
            
    # ------------------------ SMUserList required ---------------------------#

    @staticmethod
    def _identity():
        return np.zeros((6,))

    def _import(self, value, check=True):
        if isinstance(value, np.ndarray) and self.isvalid(value, check=check):
            if value.shape == (4,4):
                # it's an se(3)
                return base.vexa(value)
            elif value.shape == (6,):
                # it's a twist vector
                return value
        raise TypeError('bad type passed')

    @property
    def shape(self):
        """
        Shape of the object's interal matrix representation

        :return: (6,)
        :rtype: tuple
        """
        return (6,)

    # ------------------------ properties ---------------------------#

    @property
    def N(self):
        """
        Dimension of the object's vector representation

        :return: dimension
        :rtype: int

        Length of the Twist vector


        Example::

            >>> x = Twist()
            >>> x.N
            3
        """
        return 3

    @property
    def v(self):
        """
        Moment vector of twist

        :return: Moment vector
        :rtype: numpy.ndarray, shape=(3,)

        ``X.v`` is a 3-vector representing the moment vector of the twist.

        For example::

            >>> t = Twist3([1, 2, 3, 4, 5, 6])
            >>> t.v
            array([1, 2, 3])
        """
        return self.data[0][:3]

    @property
    def w(self):
        """
        Direction vector of twist

        :return: Direction vector
        :rtype: numpy.ndarray, shape=(3,)

        ``X.w`` is a 3-vector representing the direction vector of the twist.

        For example::

            >>> t = Twist3([1, 2, 3, 4, 5, 6])
            >>> t.w
            array([4, 5, 6])


        """
        return self.data[0][3:6]

    # -------------------- variant constructors ----------------------------#

    @classmethod
    def R(cls, a, q, p=None):
        """
        Construct a new rotational 3D twist

        :param a: Twist axis or line of action
        :type a: 3-element array_like
        :param q: Point on the line of action
        :type q: 3-element array_like
        :param p: pitch, defaults to None
        :type p: float, optional
        :return: a rotational or helical twist
        :rtype: Twist instance

        A revolute twist with a line of action in the z-direction and passing
        through (1, 2, 0) would be::

            >>> Twist3.R([0, 0, 1], [1, 2, 0])
            Twist3([2, -1, -0, 0, 0, 1])

        """
        w = base.unitvec(base.getvector(a, 3))
        v = -np.cross(w, base.getvector(q, 3))
        if p is not None:
            pitch = base.getvector(p, 3)
            v = v + pitch * w
        return cls(v, w)

    @classmethod
    def P(cls, a):
        """
        Construct a new prismatic 3D twist

        :param a: Twist axis or line of action
        :type a: 3-element array_like
        :return: a prismatic twist
        :rtype: Twist instance

        A prismatic twist with a line of action in the z-direction would be::

            >>> Twist3.P([0, 0, 1])
            Twist3([0, 0, 1, 0, 0, 0])

        """
        w = np.r_[0, 0, 0]
        v = base.unitvec(base.getvector(a, 3))

        return cls(v, w)

    # ------------------------- static methods -------------------------------#

    @staticmethod
    def isvalid(v, check=True):
        """
        Test if matrix is valid twist

        :param x: array to test
        :type x: numpy.ndarray
        :return: true of the matrix is a 6-vector or a 4x4 se(3) element
        :rtype: bool

        A twist can be reprented by a 6-vector or a 4x4 skew symmetric matrix,
        for example::

            Twist3.isvalid([1, 2, 3, 4, 5, 6])
            >>> a = base.skewa([1, 2, 3, 4, 5, 6])
            >>> a
            array([[ 0., -6.,  5.,  1.],
                [ 6.,  0., -4.,  2.],
                [-5.,  4.,  0.,  3.],
                [ 0.,  0.,  0.,  0.]])
            >>> Twist3.isvalid(a)
            True
            >>> b=np.random.rand(4,4)
            >>> Twist3.isvalid(b)
            False
        """
        if base.isvector(v, 6):
            return True
        elif base.ismatrix(v, (4, 4)):
            # maybe be an se(3)
            if not all(v.diagonal() == 0):  # check diagonal is zero
                return False
            if not all(v[3, :] == 0):  # check bottom row is zero
                return False
            if not base.isskew(v[:3, :3]):
                # top left 3x3 is skew symmetric
                return False
            return True
        return False

    # -------------------------  methods -------------------------------#

    def ad(self):
        """
        Logarithm of adjoint of 3D twist

        :return: logarithm of adjoint matrix
        :rtype: numpy.ndarray, shape=(6,6)

        ``t.ad()`` is the 6x6 logarithm of the adjoint matrix of the
          corresponding homogeneous transformation.

        For a twist representing motion from frame {B} to {A}, the adjoint will
        transform a twist relative to frame {A} to one relative to frame {B}.

        .. notes::
          
          - An alternative path to the adjoint is to exponentiate this 6x6
            matrix.
        :seealso: :func:`Twist3.Ad`
        """
        return np.array([base.skew(self.w), base.skew(self.v), [np.zeros((3, 3)), base.skew(self.w)]])

    def Ad(self):
        """
        Adjoint of 3D twist

        :return: adjoint matrix
        :rtype: numpy.ndarray, shape=(6,6)

        ``X.Ad()``  is the 6x6 adjoint matrix of the corresponding
          homogeneous transformation.

        For a twist representing motion from frame {B} to {A}, the adjoint will
        transform a twist relative to frame {A} to one relative to frame {B}.

        .. notes::
          
          - This method computes the equivalent SE(3) matrix, then the adjoint
            of that.

        :seealso: :func:`Twist3.ad` :func:`Twist3.SE3`
        """
        return self.SE3().Ad()

    def SE3(self):
        """
        Convert 3D twist to SE(3) matrix

        :return: an SE(3) representation
        :rtype: SE3 instance

        ``X.SE3()`` is an SE3 object representing the homogeneous transformation 
        equivalent to the Twist3.

        This is the exponentiation of the twist.
        """
        return SE3(self.exp())

    def se3(self):
        """
        Convert 3D twist to se(3)

        :return: An se(3) matrix
        :rtype: numpy.ndarray, shape=(4,4)

        ``X.se3()`` is the twist as an se(3) matrix, which is an augmented
        skew-symmetric 4x4 matrix.
        """
        if len(self) == 1:
            return base.skewa(self.S)
        else:
            return [base.skewa(x.S) for x in self]

    def pitch(self):
        """
        Pitch of a 3D twist

        :return: the pitch of the twist
        :rtype: float

        ``X.pitch()`` is the pitch of the twist as a scalar in units of distance
        per radian. 
        
        If we consider the twist as a screw, this is the distance of
        translation along the screw axis for a one radian rotation about the
        screw axis.
        """
        return np.dot(self.w, self.v)

    def line(self):
        """
        Line of action of 3D twist as a Plucker line

        :return: the 3D line of action
        :rtype: Plucker instance


        ``X.line()`` is a Plucker object representing the line of the twist axis.
        """
        return Plucker([Plucker(-tw.v - tw.pitch() * tw.w, tw.w) for tw in self])

    def pole(self):
        """
        Pole of a 3D twist

        :return: the pole of the twist
        :rtype: numpy.ndarray, shape=(3,)

        ``X.pole()`` is a point on the twist axis. For a pure translation 
        this point is at infinity.
        """
        return np.cross(self.w, self.v) / self.theta()

    def theta(self):
        """
        Twist rotation

        :return: rotation about the twist axis
        :rtype: float

        ``X.theta`` is the rotation about the twist axis in units of radians.

        If we consider the twist as a screw, this is the rotation about the
        screw axis to achieve the rigid-body motion.
        """
        return base.norm(self.w)

    # ------------------------- arithmetic -------------------------------#

    def __mul__(left, right):  # pylint: disable=no-self-argument
        """
        Overloaded ``*`` operator

        :arg left: left multiplicand
        :arg right: right multiplicand
        :return: product
        :raises: ValueError

        Twist composition or scaling:

        - ``X * Y`` compounds the twists ``X`` and ``Y``
        - ``X * s`` performs elementwise multiplication of the elements of ``X`` by ``s``
        - ``s * X`` performs elementwise multiplication of the elements of ``X`` by ``s``

        ========  ====================  ===================  ========================
                   Multiplicands                   Product
        -------------------------------   -----------------------------------
        left       right                type                 operation
        ========  ====================  ===================  ========================
        Twist      Twist                Twist                product of exponentials
        Twist      scalar               Twist                element-wise product
        scalar     Twist                Twist                element-wise product
        Twist      SE3                  Twist                exponential x SE3
        Twist      SpatialVelocity      SpatialVelocity      adjoint product
        Twist      SpatialAcceleration  SpatialAcceleration  adjoint product
        Twist      SpatialForce         SpatialForce         adjoint product
        ========  ====================  ===================  ========================

        Notes:

        #. Pose is ``SO2``, ``SE2``, ``SO3`` or ``SE3`` instance
        #. N is 2 for ``SO2``, ``SE2``; 3 for ``SO3`` or ``SE3``
        #. scalar x Pose is handled by ``__rmul__``
        #. scalar multiplication is commutative but the result is not a group
           operation so the result will be a matrix
        #. Any other input combinations result in a ValueError.

        For pose composition the ``left`` and ``right`` operands may be a sequence

        =========   ==========   ====  ================================
        len(left)   len(right)   len     operation
        =========   ==========   ====  ================================
         1          1             1    ``prod = left * right``
         1          M             M    ``prod[i] = left * right[i]``
         N          1             M    ``prod[i] = left[i] * right``
         M          M             M    ``prod[i] = left[i] * right[i]``
        =========   ==========   ====  ================================

        """
        # TODO TW * T compounds a twist with an SE2/3 transformation

        if isinstance(right, Twist3):
            # twist composition -> Twist
            return Twist3(left.binop(right, lambda x, y: base.trlog(base.trexp(x) @ base.trexp(y), twist=True)))
        elif isinstance(right, SE3):
            # twist * SE3 -> SE3
            return SE3(left.binop(right, lambda x, y: base.trexp(x) @ y), check=False)
        elif base.isscalar(right):
            # return Twist(left.S * right)
            return Twist3(left.binop(right, lambda x, y: x * y))
        elif isinstance(right, SpatialVelocity):
            return SpatialVelocity(left.Ad @ right.V)
        elif isinstance(right, SpatialAcceleration):
            return SpatialAcceleration(left.Ad @ right.V)
        elif isinstance(right, SpatialForce):
            return SpatialForce(left.Ad @ right.V)
        else:
            raise ValueError('twist *, incorrect right operand')

    def __rmul(right, left):  # pylint: disable=no-self-argument
        """
        Overloaded ``*`` operator

        :arg right: right multiplicand
        :arg left: left multiplicand
        :return: product
        :raises: NotImplemented

        Left-multiplication by a scalar

        - ``s * X`` performs elementwise multiplication of the elements of ``X`` by ``s``
        """
        if base.isscalar(left):
            return Twist3(self.S * left)
        else:
            raise ValueError('twist *, incorrect left operand')

    def exp(self, theta=None, units='rad'):
        """
        Exponentiate a 3D twist

        :param theta: DESCRIPTION, defaults to None
        :type theta: TYPE, optional
        :param units: DESCRIPTION, defaults to 'rad'
        :type units: TYPE, optional
        :return: homogeneous transformation
        :rtype: SE3

        -``X.exp()`` is the homogeneous transformation equivalent to the twist.
        -``X.exp(θ) as above but with a rotation of ``θ`` about the twist axis.

        .. notes::

            - For the second form, the twist must, if rotational, have a unit 
              rotational component.

        See also Twist3.T, trexp, trexp2.
        """
        if units != 'rad' and self.isprismatic:
            print('Twist3.exp: using degree mode for a prismatic twist')

        if theta is None:
            theta = 1
        else:
            theta = base.getunit(theta, units)

        if base.isscalar(theta):
            # theta is a scalar
            return SE3(base.trexp(self.S * theta))
        else:
            # theta is a vector
            if len(self) == 1:
                return SE3([base.trexp(self.S * t) for t in theta])
            elif len(self) == len(theta):
                return SE3([base.trexp(S * t) for S, t in zip(self.data, theta)])
            else:
                raise ValueError('length of twist and theta not consistent')

    def prod(self):
        r"""
        Product of 3D twists
 
        :return: Product of elements
        :rtype: Twist3

        For a twist instance with N values return the matrix product of those
        elements :math:`\prod_i^N S_i`.
        """
        twprod = base.trexp(self.data[0])

        for tw in self.data[1:]:
            twprod = twprod @ base.trexp(tw)
        return Twist3(base.trlog(twprod))

    def __str__(self):
        """
        Pretty string representation of 3D twist

        :return: readable representation of the twist
        :rtype: str

        Convert the twist's value to an array of numbers.

        Example::

            >>> x = Twist3.R([1,2,3], [4,5,6])
            >>> print(x)
            (0.80178 -1.6036 0.80178; 0.26726 0.53452 0.80178)
        """
        return '\n'.join(["({:.5g} {:.5g} {:.5g}; {:.5g} {:.5g} {:.5g})".format(*list(base.removesmall(tw.S))) for tw in self])

    def __repr__(self):
        """
        Readable representation of 3D twist

        :return: readable representation of a twist as a list of arrays
        :rtype: str

        Example::

            >>> x = Twist3.R([1,2,3], [4,5,6])
            >>> x
            Twist3([0.80178, -1.6036, 0.80178, 0.26726, 0.53452, 0.80178])
            >>> a.append(a)
            >>> a
            Twist3([
              [0.80178, -1.6036, 0.80178, 0.26726, 0.53452, 0.80178],
              [0.80178, -1.6036, 0.80178, 0.26726, 0.53452, 0.80178]
            ])

        """
        if len(self) == 0:
            return "Twist([])"
        elif len(self) == 1:
            return "Twist3([{:.5g}, {:.5g}, {:.5g}, {:.5g}, {:.5g}, {:.5g}])".format(*list(self.S))
        else:
            return "Twist3([\n" + \
                ',\n'.join(["  [{:.5g}, {:.5g}, {:.5g}, {:.5g}, {:.5g}, {:.5g}]".format(*list(tw)) for tw in self.data]) +\
                "\n])"

    def _repr_pretty_(self, p, cycle):
        """
        Pretty string for IPython

        :param p: pretty printer handle (ignored)
        :param cycle: pretty printer flag (ignored)

        Print colorized output when variable is displayed in IPython, ie. on a line by
        itself.

        Example::

            In [1]: x

        """
        p.begin_group(8, 'Twist3(')
        for i, x in enumerate(self):
            p.break_()
            p.text(str(x))
        p.end_group(8, ')')

# ======================================================================== #


class Twist2(SMTwist):
    def __init__(self, arg=None, w=None, check=True):
        """
        Construct a new 2D Twist object

        :type a: 2-element array-like
        :return: 2D prismatic twist
        :rtype: Twist2 instance

        - ``Twist2(R)`` is a 2D Twist object representing the SO(2) rotation expressed as 
          a 2x2 matrix.
        - ``Twist2(T)`` is a 2D Twist object representing the SE(2) rigid-body motion expressed as 
          a 3x3 matrix.
        - ``Twist2(X)`` if X is an SO2 instance then create a 2D Twist object representing the SO(2) rotation,
          and if X is an SE2 instance then create a 2D Twist object representing the SE(2) motion
        - ``Twist2(V)`` is a  2D Twist object specified directly by a 3-element array-like comprising the
          moment vector (1 element) and direction vector (2 elements).
        """

        if w is None:
            # zero or one arguments passed
            if super().arghandler(arg, convertfrom=(SE2,), check=check):
                return

            elif base.isvector(arg, 6):
                # Twist(array_like)
                self.data = [base.getvector(arg)]
                return

        elif w is not None and base.isvector(w, 1) and base.isvector(arg,2):
            # Twist(v, w)
            self.data = [np.r_[arg, w]]
            return

        raise ValueError('bad twist value')

    @staticmethod
    def _identity():
        return np.zeros((3,))

    @property
    def shape(self):
        """
        Shape of the object's interal matrix representation

        :return: (3,)
        :rtype: tuple
        """
        return (3,)

    def _import(self, value, check=True):
        if isinstance(value, np.ndarray) and self.isvalid(value, check=check):
            if value.shape == (3,3):
                # it's an se(3)
                return base.vexa(value)
            elif value.shape == (3,):
                # it's a twist vector
                return value
        raise TypeError('bad type passed')
    # -------------------- variant constructors ----------------------------#

    @classmethod
    def R(cls, q):
        """
        Construct a new 2D revolute Twist object

        :param a: displacment
        :type a: 2-element array-like
        :return: 2D prismatic twist
        :rtype: Twist2 instance

        - ``Twist3.R(q)`` is a 2D Twist object representing rotation about the 2D point ``q``.
        """

        q = base.getvector(q, 2)
        v = -np.cross(np.r_[0.0, 0.0, 1.0], np.r_[q, 0.0])
        return cls(v[:2], 1)

    @classmethod
    def P(cls, a):
        """
        Construct a new 2D primsmatic Twist object

        :param a: displacment
        :type a: 2-element array-like
        :return: 2D prismatic twist
        :rtype: Twist2 instance

        - ``Twist3.P(q)`` is a 2D Twist object representing 2D-translation in the direction ``a``.
        """
        w = 0
        v = base.unitvec(base.getvector(a, 2))
        return cls(v, w)

    @property
    def N(self):
        """
        Dimension of the object's vector representation

        :return: dimension
        :rtype: int

        Length of the Twist vector


        Example::

            >>> x = Twist2()
            >>> x.N
            3
        """
        return 2

    @property
    def v(self):
        """
        Twist as a moment vector

        :return: Moment vector
        :rtype: numpy.ndarray, shape=(2,)

        - ``X.v`` is a 2-vector

        """
        return self.data[0][:2]

    @property
    def w(self):
        """
        Twist as a direction vector

        :return: Direction vector
        :rtype: float

        - ``X.v`` is a 2-vector

        """
        return self.data[0][2]

    # ------------------------- static methods -------------------------------#

    @staticmethod
    def isvalid(v, check=True):
        if base.isvector(v, 3):
            return True
        elif base.ismatrix(v, (3, 3)):
            # maybe be an se(2)
            if not all(v.diagonal() == 0):  # check diagonal is zero
                return False
            if not all(v[2, :] == 0):  # check bottom row is zero
                return False
            if not base.isskew(v[:2, :2]):
                # top left 2x2is skew symmetric
                return False
            return True
        return False

    def SE2(self):
        """
        %Twist3.SE Convert twist to SE2 or SE3 object
        %
        TW.SE is an SE2 or SE3 object representing the homogeneous transformation equivalent to the Twist3.
                %
            See also Twist3.T, SE2, SE3.
        """

        return SE2(self.exp())

    def se2(self):
        """
        Twist3.se Return the twist matrix

        TW.se is the twist matrix in se(2) or se(3) which is an augmented
        skew-symmetric matrix (3x3 or 4x4).

        """
        if len(self) == 1:
            return base.skewa(self.S)
        else:
            return [base.skewa(x.S) for x in self]

    def exp(self, theta=None, units='rad'):
        """
        Twist3.exp Convert twist to homogeneous transformation

        TW.exp is the homogeneous transformation equivalent to the twist (SE2 or SE3).

        TW.exp(THETA) as above but with a rotation of THETA about the Twist3.

        Notes::
        - For the second form the twist must, if rotational, have a unit rotational component.

        See also Twist3.T, trexp, trexp2.
        """

        if units != 'rad' and self.isprismatic:
            print('Twist3.exp: using degree mode for a prismatic twist')

        if theta is None:
            theta = 1
        else:
            theta = base.getunit(theta, units)

        if base.isscalar(theta):
            return SE2(base.trexp2(self.S * theta))
        else:
            return SE2([base.trexp2(self.S * t) for t in theta])

    @property
    def unit(self):
        """
        Unit twist

        TW.unit() is a Twist object representing a unit aligned with the Twist
        TW.
        """
        if base.iszerovec(self.w):
            # rotational twist
            return Twist2(self.S / base.norm(S.w))
        else:
            # prismatic twist
            return Twist2(base.unitvec(self.v), [0, 0, 0])

    @property
    def ad(self):
        """
        Twist3.ad Logarithm of adjoint

        TW.ad is the logarithm of the adjoint matrix of the corresponding
        homogeneous transformation.

        See also SE3.Ad.
        """
        return np.array([base.skew(self.w), base.skew(self.v), [np.zeros((3, 3)), base.skew(self.w)]])

    def __mul__(self, right):
        """
        Twist3.mtimes Multiply twist by twist or scalar

        TW1 * TW2 is a new Twist representing the composition of twists TW1 and
        TW2.

        TW * T is an SE2 or SE3 that is the composition of the twist TW and the
        homogeneous transformation object T.

        TW * S with its twist coordinates scaled by scalar S.

        TW * T compounds a twist with an SE2/3 transformation
        %
        """
        left = self
        if isinstance(right, Twist2):
            # twist composition
            return Twist2(left.exp() * right.exp())
        elif base.isscalar(right):
            return Twist2(left.S * right)
        else:
            raise ValueError('twist *, incorrect right operand')

    def __rmul(self, left):
        if base.isscalar(left):
            return Twist2(self.S * left)
        else:
            raise ValueError('twist *, incorrect left operand')

    def prod(self):
        """
        %Twist3.prod Compound array of twists
        %
        TW.prod is a twist representing the product (composition) of the
        successive elements of TW (1xN), an array of Twists.
                %
                %
        See also RTBPose.prod, Twist3.mtimes.
        """
        twprod = base.trexp2(self.data[0])

        for tw in self.data[1:]:
            twprod = twprod @ base.trexp2(tw)
        return Twist2(base.trlog2(twprod, twist=True))

    def __str__(self):
        """
    %Twist3.char Convert to string

    s = TW.char() is a string showing Twist parameters in a compact single line format.
    If TW is a vector of Twist objects return a string with one line per Twist3.

    See also Twist3.display.
        """
        return '\n'.join(["({:.5g} {:.5g}; {:.5g})".format(*list(tw.S)) for tw in self])

    def __repr__(self):
        """
        %Twist3.display Display parameters
        %
L.display() displays the twist parameters in compact single line format.  If L is a
vector of Twist objects displays one line per element.
        %
Notes::
- This method is invoked implicitly at the command line when the result
  of an expression is a Twist object and the command has no trailing
  semicolon.
        %
See also Twist3.char.
        """

        if len(self) == 1:
            return "Twist2([{:.5g}, {:.5g}, {:.5g}])".format(*list(self.S))
        else:
            return "Twist2([\n" + \
                ',\n'.join(["  [{:.5g}, {:.5g}, {:.5g}}]".format(*list(tw.S)) for tw in self]) +\
                "\n])"

    def _repr_pretty_(self, p, cycle):
        """
        Pretty string for IPython

        :param p: pretty printer handle (ignored)
        :param cycle: pretty printer flag (ignored)

        Print colorized output when variable is displayed in IPython, ie. on a line by
        itself.

        Example::

            In [1]: x

        """
        p.begin_group(8, 'Twist2(')
        for i, x in enumerate(self):
            p.break_()
            p.text(str(x))
        p.end_group(8, ')')

if __name__ == '__main__':   # pragma: no cover

    import pathlib

    a = Twist3()

    exec(open(pathlib.Path(__file__).parent.absolute() / "test_twist.py").read())