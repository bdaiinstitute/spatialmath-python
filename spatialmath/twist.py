import numpy as np
import math

from spatialmath.base import argcheck
import spatialmath.base as tr
from spatialmath import super_pose as sp
from spatialmath import SO3, SE3, SO2, SE2, Plucker
from spatialmath.smuserlist import SMUserList


class SMTwist(SMUserList):
    """
    Superclass for 2D and 3D twist objects

    Subclasses are:

    - ``Twist2`` representing rigid-body motion in 2D as a 3-vector
    - ``Twist`` representing rigid-body motion in 3D as a 6-vector

    A twist is the unique elements of the logarithm of the corresponding SE(N)
    matrix.

    Arithmetic operators are overloaded but the operation they perform depend
    on the types of the operands.  For example:

    - ``*`` will compose two instances of the same subclass, and the result will be
      an instance of the same subclass, since this is a group operator.

    These classes all inherit from ``UserList`` which enables them to 
    represent a sequence of values, ie. an ``Twist`` instance can contain
    a sequence of twists.  Most of the Python ``list`` operators
    are applicable::

        >>> x = Twist()  # new instance with zero value
        >>> len(x)     # it is a sequence of one value
        1
        >>> x.append(x)  # append to itself
        >>> len(x)       # it is a sequence of two values
        2
        >>> x[1]         # the element has a 4x4 matrix value
        SE3([
        array([[1., 0., 0., 0.],
               [0., 1., 0., 0.],
               [0., 0., 1., 0.],
            [0., 0., 0., 1.]]) ])
        >>> x[1] = SE3.Rx(0.3)  # set an elements of the sequence
        >>> x.reverse()         # reverse the elements in the sequence
        >>> del x[1]            # delete an element

    """
    # ------------------------- list support -------------------------------#

    def __init__(self):
        # handle common cases
        #  deep copy
        #  numpy array
        #  list of numpy array
        # validity checking??
        # TODO should this be done by __new__?
        super().__init__()   # enable UserList superpowers

    @property
    def S(self):
        """
        Twist as a vector (superclass property)

        :return: Twist vector
        :rtype: numpy.ndarray, shape=(N,)

        - ``X.S`` is a 3-vector if X is a ``Twist2`` instance, and a 6-vector if
          X is a ``Twist`` instance.

        Notes::


        - the vector is the unique elements of the se(N) representation
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

        :return: If twist is purely prismatic
        :rtype: book

        Example::

            >>> x = Twist3.R([1,2,3], [4,5,6])
            >>> x.isprismatic
            False

        """
        if len(self) == 1:
            return tr.iszerovec(self.w)
        else:
            return [tr.iszerovec(x.w) for x in self.data]

    @property
    def unit(self):
        """
        Unit twist

        TW.unit() is a Twist object representing a unit aligned with the Twist
        TW.
        """
        if tr.iszerovec(self.w):
            # rotational twist
            return Twist(self.S / tr.norm(S.w))
        else:
            # prismatic twist
            return Twist(tr.unitvec(self.v), [0, 0, 0])

    @property
    def isunit(self):
        """
        Test for unit twist (superclass property)

        :return: If twist is a unit-twist
        :rtype: bool
        """
        if len(self) == 1:
            return tr.isunittwist(self.S)
        else:
            return [tr.isunittwist(x) for x in self.data]

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
        out = self[0]

        for t in self[1:]:
            out *= t
        return out

# ======================================================================== #


class Twist3(SMTwist):
    """
    TWIST SE(2) and SE(3) Twist class

    A Twist class holds the parameters of a twist, a representation of a
    rigid body displacement in SE(2) or SE(3).

    Methods::
     S             twist vector (1x3 or 1x6)
     se            twist as (augmented) skew-symmetric matrix (3x3 or 4x4)
     T             convert to homogeneous transformation (3x3 or 4x4)
     R             convert rotational part to matrix (2x2 or 3x3)
     exp           synonym for T
     ad            logarithm of adjoint
     pitch         pitch of the screw, SE(3) only
     pole          a point on the line of the screw
     prod          product of a vector of Twists
     theta         rotation about the screw
     line          Plucker line object representing line of the screw
     display       print the Twist parameters in human readable form
     char          convert to string

    Conversion methods::
     SE            convert to SE2 or SE3 object
     double        convert to real vector

    Overloaded operators::
     *             compose two Twists
     *             multiply Twist by a scalar

    Properties (read only)::
     v             moment part of twist (2x1 or 3x1)
     w             direction part of twist (1x1 or 3x1)

    References::
    - "Mechanics, planning and control"
      Park & Lynch, Cambridge, 2016.

    See also trexp, trexp2, trlog.

    Copyright (C) 1993-2019 Peter I. Corke

    This file is part of The Spatial Math Toolbox for Python (SMTB-P)

    https://github.com/petercorke/spatial-math
    """

    def __init__(self, arg=None, w=None, check=True):
        """
        Construct a new Twist object

        TW = Twist(T) is a Twist object representing the SE(2) or SE(3)
        homogeneous transformation matrix T (3x3 or 4x4).

        TW = Twist(V) is a twist object where the vector is specified directly.

        3D CASE:

        TW = Twist('R', A, Q) is a Twist object representing rotation about the
        axis of direction A (3x1) and passing through the point Q (3x1).
                %
        TW = Twist('R', A, Q, P) as above but with a pitch of P (distance/angle).

        TW = Twist('T', A) is a Twist object representing translation in the
        direction of A (3x1).

        Notes:

        - The argument 'P' for prismatic is synonymous with 'T'.
        """

        super().__init__()   # enable UserList superpowers

        if arg is None:
            # Twist()
            self.data = [np.r_[0, 0, 0, 0, 0, 0]]
        elif argcheck.isvector(arg, 6):
            # Twist(array_like)
            self.data = [argcheck.getvector(arg)]
        elif w is not None and argcheck.isvector(w, 3) and argcheck.isvector(arg,3):
            # Twist(v, w)
            self.data = [np.r_[arg, w]]
        else:
            super().arghandler(arg, (SE3,), check=check)

    def _import(self, value, check=True):
        if isinstance(value, np.ndarray) and self.isvalid(value, check=check):
            if value.shape == (4,4):
                # it's an se(3)
                return tr.vexa(value)
            elif value.shape == (6,):
                # it's a twist vector
                return value
        raise TypeError('bad type passed')

    # ------------------------- properties -------------------------------#
    @property
    def shape(self):
        """
        Shape of the object's interal matrix representation

        :return: (6,)
        :rtype: tuple
        """
        return (6,)

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
        Twist as a moment vector

        :return: Moment vector
        :rtype: numpy.ndarray, shape=(3,)

        - ``X.v`` is a 3-vector

        """
        return self.data[0][:3]

    @property
    def w(self):
        """
        Twist as a direction vector

        :return: Direction vector
        :rtype: numpy.ndarray, shape=(3,)

        - ``X.v`` is a 3-vector for Twist and a 1-vector for Twist2

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

        """

        w = tr.unitvec(argcheck.getvector(a, 3))
        v = -np.cross(w, argcheck.getvector(q, 3))
        if p is not None:
            pitch = argcheck.getvector(p, 3)
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

        """
        w = np.r_[0, 0, 0]
        v = tr.unitvec(argcheck.getvector(a, 3))

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

        """
        if argcheck.isvector(v, 6):
            return True
        elif argcheck.ismatrix(v, (4, 4)):
            # maybe be an se(3)
            if not all(v.diagonal() == 0):  # check diagonal is zero
                return False
            if not all(v[3, :] == 0):  # check bottom row is zero
                return False
            if not tr.isskew(v[:3, :3]):
                # top left 3x3 is skew symmetric
                return False
            return True
        return False

    # -------------------------  methods -------------------------------#

    def ad(self):
        """
        Logarithm of adjoint

        :return: logarithm of adjoint matrix
        :rtype: numpy.ndarray, shape=(6,6)

        - ``X.ad()`` is the 6x6 logarithm of the adjoint matrix of the corresponding
          homogeneous transformation.
        """
        return np.array([tr.skew(self.w), tr.skew(self.v), [np.zeros((3, 3)), tr.skew(self.w)]])

    def Ad(self):
        """
        Twist3.Ad Adjoint

        :return: adjoint matrix
        :rtype: numpy.ndarray, shape=(6,6)

        - ``X.Ad()``  is the 6x6 adjoint matrix of the corresponding
          homogeneous transformation.
        """
        return self.SE3().Ad()

    def SE3(self):
        """
        Convert twist to SE(3)

        :return: an SE(3) representation
        :rtype: SE3 instance

        - ``X.SE3()`` is an SE3 object representing the homogeneous transformation 
          equivalent to the Twist3.

        """
        return SE3(self.exp())

    def se3(self):
        """
        Convert twist to se(3)

        :return: An se(3) matrix
        :rtype: numpy.ndarray, shape=(4,4)

        - ``X.se3()`` is the twist as an se(3) matrix, which is an augmented
          skew-symmetric 4x4 matrix.
        """
        if len(self) == 1:
            return tr.skewa(self.S)
        else:
            return [tr.skewa(x.S) for x in self]

    def pitch(self):
        """
        Twist pitch

        :return: the pitch of the twist
        :rtype: float

        - ``X.pitch()`` is the pitch of the Twist as a scalar in units of distance per radian.
        """
        return np.dot(self.w, self.v)

    def line(self):
        """
        Twist line of action in Plucker form

        :return: the 3D line of action
        :rtype: Plucker instance


        - ``X.line()`` is a Plucker object representing the line of the twist axis.
        """
        tw = self
        return Plucker(tw.v - tw.pitch() * tw.w, tw.w)

    def pole(self):
        """
        Twist pole

        :return: the pole of the twist
        :rtype: numpy.ndarray, shape=(3,)

        - ``X.pole()`` is a point on the twist axis. For a pure translation 
          this point is at infinity.
        """
        return np.cross(self.w, self.v) / self.theta()

    def theta(self):
        """
        Twist rotation

        :return: rotation about the twist axis
        :rtype: float

        - ``X.theta`` is the rotation about the twist axis in units of radians.
        """
        return tr.norm(self.w)

    # ------------------------- arithmetic -------------------------------#

    def __mul__(left, right):
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
            return Twist3(left.binop(right, lambda x, y: tr.trlog(tr.trexp(x) @ tr.trexp(y), twist=True)))
        elif isinstance(right, SE3):
            # twist * SE3 -> SE3
            return SE3(left.binop(right, lambda x, y: tr.trexp(x) @ y), check=False)
        elif argcheck.isscalar(right):
            # return Twist(left.S * right)
            return Twist3(left.binop(right, lambda x, y: x * y))
        elif isinstance(right, SpatialVelocity):
            return SpatialVelocity(a.Ad @ b.vw)
        elif isinstance(right, SpatialAcceleration):
            return SpatialAcceleration(a.Ad @ b.vw)
        elif isinstance(right, SpatialForce):
            return SpatialForce(a.Ad @ b.vw)
        else:
            raise ValueError('twist *, incorrect right operand')

    def __imul__(left, right):
        return left.__mul__(right)

    def __rmul(right, left):
        if argcheck.isscalar(left):
            return Twist3(right.S * left)
        else:
            raise ValueError('twist *, incorrect left operand')

    def exp(self, theta=None, units='rad'):
        """
        Exponentiate a twist

        :param theta: DESCRIPTION, defaults to None
        :type theta: TYPE, optional
        :param units: DESCRIPTION, defaults to 'rad'
        :type units: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE

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
            theta = argcheck.getunit(theta, units)

        if isinstance(theta, (int, np.int64, float, np.float64)):
            return SE3(tr.trexp(self.S * theta))
        else:
            return SE3([tr.trexp(self.S * t) for t in theta])

    def __str__(self):
        """
        Pretty string representation of twist

        :return: readable representation of the twist
        :rtype: str

        Convert the twist's value to an array of numbers.

        Example::

            >>> x = Twist3.R([1,2,3], [4,5,6])
            >>> print(x)
            (0.80178 -1.6036 0.80178; 0.26726 0.53452 0.80178)

        Notes:

            - By default, the output is colorised for an ANSI terminal console:

                * red: rotational elements
                * blue: translational elements
                * white: constant elements

        """
        return '\n'.join(["({:.5g} {:.5g} {:.5g}; {:.5g} {:.5g} {:.5g})".format(*list(tw.S)) for tw in self])

    def __repr__(self):
        """
        Readable representation of twist

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
        Pretty string for IPython (superclass method)

        :param p: pretty printer handle (ignored)
        :param cycle: pretty printer flag (ignored)

        Print colorized output when variable is displayed in IPython, ie. on a line by
        itself.

        Example::

            In [1]: x

        """
        print(self.__str__())

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

        super().__init__()   # enable UserList superpowers

        if arg is None:
            self.data = [np.r_[0.0, 0.0, 0.0, ]]

        elif isinstance(arg, Twist2):
            # clone it
            self.data = [np.r_[arg.v, arg.w]]

        elif argcheck.isvector(arg, 3):
            s = argcheck.getvector(arg)
            self.data = [s]

        elif argcheck.isvector(arg, 2) and argcheck.isvector(w, 1):
            v = argcheck.getvector(arg)
            w = argcheck.getvector(w)
            self.data = [np.r_[v, w]]

        elif isinstance(arg, SE2):
            S = tr.trlog2(arg.A)  # use closed form for SE(2)

            skw, v = tr.tr2rt(S)
            w = tr.vex(skw)
            self.data = [np.r_[v, w]]

        elif Twist2.isvalid(arg):
            # it's an augmented skew matrix, unpack it
            skw, v = tr.tr2rt(arg)
            w = tr.vex(skw)
            self.data = [np.r_[v, w]]

        elif isinstance(arg, list):
            # construct from a list

            if isinstance(arg[0], np.ndarray):
                # possibly a list of numpy arrays
                if check:
                    assert all(map(lambda x: Twist2.isvalid(x), arg)), 'all elements of list must have valid shape and value for the class'
                self.data = arg
            elif type(arg[0]) == type(self):
                # possibly a list of objects of same type
                assert all(map(lambda x: type(x) == type(self), arg)), 'all elements of list must have same type'
                self.data = [x.S for x in arg]
            elif type(arg[0]) == list:
                # possibly a list of 3-lists
                assert all(map(lambda x: isinstance(x, list) and len(x) == 3, arg)), 'all elements of list must have same type'
                self.data = [np.r_[x] for x in arg]
            else:
                raise ValueError('bad list argument to constructor')

        else:
            raise ValueError('bad argument to constructor')

    @property
    def shape(self):
        """
        Shape of the object's interal matrix representation

        :return: (3,)
        :rtype: tuple
        """
        return (3,)

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

        q = argcheck.getvector(q, 2)
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
        v = tr.unitvec(argcheck.getvector(a, 2))
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
        if argcheck.isvector(v, 3):
            return True
        elif argcheck.ismatrix(v, (3, 3)):
            # maybe be an se(2)
            if not all(v.diagonal() == 0):  # check diagonal is zero
                return False
            if not all(v[2, :] == 0):  # check bottom row is zero
                return False
            if not tr.isskew(v[:2, :2]):
                # top left 2x2is skew symmetric
                return False
            return True
        return False

    @property
    def SE2(tw):
        """
        %Twist3.SE Convert twist to SE2 or SE3 object
        %
        TW.SE is an SE2 or SE3 object representing the homogeneous transformation equivalent to the Twist3.
                %
            See also Twist3.T, SE2, SE3.
        """

        return SE2(tw.exp())

    @property
    def se2(self):
        """
        Twist3.se Return the twist matrix

        TW.se is the twist matrix in se(2) or se(3) which is an augmented
        skew-symmetric matrix (3x3 or 4x4).

        """
        if len(self) == 1:
            return tr.skewa(self.S)
        else:
            return [tr.skewa(x.S) for x in self]

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
            theta = argcheck.getunit(theta, units)

        if isinstance(theta, (int, np.int64, float, np.float64)):
            return SE2(tr.trexp2(self.S * theta))
        else:
            return SE2([tr.trexp2(self.S * t) for t in theta])

    @property
    def unit(self):
        """
        Unit twist

        TW.unit() is a Twist object representing a unit aligned with the Twist
        TW.
        """
        if tr.iszerovec(self.w):
            # rotational twist
            return Twist2(self.S / tr.norm(S.w))
        else:
            # prismatic twist
            return Twist2(tr.unitvec(self.v), [0, 0, 0])

    @property
    def ad(self):
        """
        Twist3.ad Logarithm of adjoint

        TW.ad is the logarithm of the adjoint matrix of the corresponding
        homogeneous transformation.

        See also SE3.Ad.
        """
        x = np.array([skew(self.w), skew(self.v), [np.zeros((3, 3)), skew(self.w)]])

    def __mul__(left, right):
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

        if isinstance(right, Twist2):
            # twist composition
            return Twist2(left.exp() * right.exp())
        elif isinstance(right, (int, np.int64, float, np.float64)):
            return Twist2(left.S * right)
        else:
            raise ValueError('twist *, incorrect right operand')

    def __imul__(left, right):
        return left.__mul__(right)

    def __rmul(right, left):
        if isinstance(left, (int, np.int64, float, np.float64)):
            return Twist2(right.S * left)
        else:
            raise ValueError('twist *, incorrect left operand')

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
        Pretty string for IPython (superclass method)

        :param p: pretty printer handle (ignored)
        :param cycle: pretty printer flag (ignored)

        Print colorized output when variable is displayed in IPython, ie. on a line by
        itself.

        Example::

            In [1]: x

        """
        # note _repr_pretty_self must be in subclass if the subclass has a __repr__
        #  see https://stackoverflow.com/questions/59284262/how-to-implement-repr-pretty-in-a-python-mixin
        print(self.__str__())

if __name__ == '__main__':   # pragma: no cover

    import pathlib



    # exec(open(pathlib.Path(__file__).parent.absolute() / "test_twist.py").read())
