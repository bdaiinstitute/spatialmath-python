
"""
Spatial vectors

A set of cooperating classes to support Featherstone's spatial vector formalism

References::

 - Robot Dynamics Algorithms, R. Featherstone, volume 22,
   Springer International Series in Engineering and Computer Science,
   Springer, 1987.
 - A beginner's guide to 6-d vectors (part 1), R. Featherstone,
   IEEE Robotics Automation Magazine, 17(3):83-94, Sep. 2010.


     SpatialVec6 (abstract super class)
       |
       +--- SpatialM6 (abstract)
       |     |
       |     +---SpatialVelocity
       |     +---SpatialAcceleration
       |
       +---SpatialF6 (abstract)
            |
            +---SpatialForce
            +---SpatialMomentum
"""

from abc import ABC, abstractmethod
from collections import UserList
import spatialmath.base.argcheck as arg
import spatialmath.base as tr
import numpy as np

class SpatialVector(UserList, ABC):
    """
        SpatialV6 Abstract spatial 6-vector class

        Abstract superclass for spatial vector functionality.  This class has two
        abstract subclasses, which each have concrete subclasses:


        Methods::
         SpatialV6     constructor invoked by subclasses
         double        convert to a 6xN double
         char          convert to string
         display       display in human readable form

        Operators::

         +          add spatial vectors of the same type
         -          subtract spatial vectors of the same type
         -          unary minus of spatial vectors

        Notes::
        - Subclass of the MATLAB handle class which means that pass by reference semantics
          apply.
        - Spatial vectors can be placed into arrays and indexed.

        See also SpatialM6, SpatialF6, SpatialVelocity, SpatialAcceleration, SpatialForce,
        SpatialMomentum, SpatialInertia.
    """

    @abstractmethod
    def __init__(self, value):
        """
        Create a new spatial vector (abstract superclass)

        :param value: Value of the

        - ``SpatialVector(vec)`` is a spatial vector constructed from the 6-element array-like ``vec``
        - ``SpatialVector([V1, V2, ... VN])`` is a spatial vector array with N elements, constructed from the 6-element
           array-like values ``Vi``
        - ``SpatialVector(A)`` is a spatial vector array with N elements, constructed from the columns of the 6xN
          array ``A``.

        See also SpatialVelocity, SpatialAcceleration, SpatialForce, SpatialMomentum.
        """
        print('spatialVec6 init')
        super().__init__()

        if value is None:
            self.data = [np.zeros((6,))]
        elif arg.isvector(value, 6):
            self.data = [np.array(value)]
        elif isinstance(value, list):
            assert all(map(lambda x: arg.isvector(x, 6), value)), 'all elements of list must have valid shape and value for the class'
            self.data = [np.array(x) for x in value]
        elif arg.ismatrix(value, (6, None)):
            self.data = [x for x in value.T]
        else:
            raise ValueError('bad arguments to constructor')

    @classmethod
    def Empty(cls):
        """
        Construct a new spatial vector with zero items (superclass method)

        :param cls: The spatial vector subclass
        :type cls: SpatialVelocity, SpatialAcceleration, SpatialForce
        :return: a spatial vector array with no values
        :rtype: SpatialVelocity, SpatialAcceleration, SpatialForce or SpatialMomentum instance

        This constructs an empty pose container which can be appended to.  For example::

            >>> x = SpatialVelocity.Empty()
            >>> len(x)
            0
            >>> x.append(SpatialVelocity([1,2,3,4,5,6)
            >>> len(x)
            1

        """
        X = cls()
        X.data = []
        return X

        # ------------------------------------------------------------------------ #

    def __getitem__(self, i):
        """
        Access value of a spatial object (superclass method)

        :param i: index of element to return
        :type i: int
        :return: the specific element of the pose
        :rtype: SO2, SE2, SO3, SE3 instance
        :raises IndexError: if the element is out of bounds

        Note that only a single index is supported, slices are not.

        Example::

            >>> x = SE3.Rx([0, math.pi/2, math.pi])
            >>> len(x)
            3
            >>> x[1]
               1           0           0           0
               0           0          -1           0
               0           1           0           0
               0           0           0           1
        """
        print('getitem', i)
        if isinstance(i, slice):
            return self.__class__([self.data[k] for k in range(i.start or 0, i.stop or len(self), i.step or 1)])
        else:
            return self.__class__(self.data[i])

    def __setitem__(self, i, value):
        """
        Assign a value to a pose object (superclass method)

        :param i: index of element to assign to
        :type i: int
        :param value: the value to insert
        :type value: SO2, SE2, SO3, SE3 instance
        :raises ValueError: incorrect type of assigned value

        Assign the argument to an element of the object's internal list of values.
        This supports the assignement operator, for example::

            >>> x = SE3([SE3() for i in range(10)]) # sequence of ten identity values
            >>> len(x)
            10
            >>> x[3] = SE3.Rx(0.2)   # assign to position 3 in the list
        """
        if not type(self) == type(value):
            raise ValueError("cant append different type of pose object")
        if len(value) > 1:
            raise ValueError("cant insert a pose sequence - must have len() == 1")
        self.data[i] = value.A

    def append(self, x):
        """
        Append a value to a pose object (superclass method)

        :param x: the value to append
        :type x: SO2, SE2, SO3, SE3 instance
        :raises ValueError: incorrect type of appended object

        Appends the argument to the object's internal list of values.

        Examples::

            >>> x = SE3()
            >>> len(x)
            1
            >>> x.append(SE3.Rx(0.1))
            >>> len(x)
            2
        """
        print('in append method')
        if not type(self) == type(x):
            raise ValueError("cant append different type of spatial vector")
        if len(x) > 1:
            raise ValueError("cant append a spatial vector sequence - use extend")
        super().append(x.V)

    def extend(self, x):
        """
        Extend sequence of values of a pose object (superclass method)

        :param x: the value to extend
        :type x: SO2, SE2, SO3, SE3 instance
        :raises ValueError: incorrect type of appended object

        Appends the argument to the object's internal list of values.

        Examples::

            >>> x = SE3()
            >>> len(x)
            1
            >>> x.append(SE3.Rx(0.1))
            >>> len(x)
            2
        """
        # print('in extend method')
        if not type(self) == type(x):
            raise ValueError("cant append different type of pose object")
        if len(x) == 0:
            raise ValueError("cant extend a singleton pose  - use append")
        super().extend(x.A)

    def insert(self, i, value):
        """
        Insert a value to a pose object (superclass method)

        :param i: element to insert value before
        :type i: int
        :param value: the value to insert
        :type value: SO2, SE2, SO3, SE3 instance
        :raises ValueError: incorrect type of inserted value

        Inserts the argument into the object's internal list of values.

        Examples::

            >>> x = SE3()
            >>> x.insert(0, SE3.Rx(0.1)) # insert at position 0 in the list
            >>> len(x)
            2
        """
        if not type(self) == type(value):
            raise ValueError("cant append different type of pose object")
        if len(value) > 1:
            raise ValueError("cant insert a pose sequence - must have len() == 1")
        super().insert(i, value.V)

    def pop(self):
        """
        Pop value of a pose object (superclass method)

        :return: the specific element of the pose
        :rtype: SO2, SE2, SO3, SE3 instance
        :raises IndexError: if there are no values to pop

        Removes the first pose value from the sequence in the pose object.

        Example::

            >>> x = SE3.Rx([0, math.pi/2, math.pi])
            >>> len(x)
            3
            >>> y = x.pop()
            >>> y
            SE3(array([[ 1.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00],
                       [ 0.0000000e+00, -1.0000000e+00, -1.2246468e-16,  0.0000000e+00],
                       [ 0.0000000e+00,  1.2246468e-16, -1.0000000e+00,  0.0000000e+00],
                       [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]]))
            >>> len(x)
            2
        """

        return self.__class__(super().pop())

    # ------------------------------------------------------------------------ #
    @property
    def V(self):
        """
        Spatial vector as an array

        :return: Moment vector
        :rtype: numpy.ndarray, shape=(3,)

        - ``X.v`` is a 3-vector

        """
        return self.data[0]

    def __repr__(self):
        """

        :return:
        SpatialVec6.display Display parameters

        V.display() displays the spatial vector parameters in compact single line format.
        If V is an array of spatial vector objects it displays one per line.

        Notes:

         - This method is invoked implicitly at the command line when the result
           of an expression is a serial vector subclass object and the command has
           no trailing semicolon.
        """
        return self.__str__()

    def __str__(self):
        """
        Pretty string representation of spatial vector

        :return: readable representation of the spatial vector
        :rtype: str

        - ``s = str(v)`` is a string showing spatial vector parameters in a
        compact single line format.

        If V is an array of spatial vector objects return a string with one
        line per element.
        """
        typ = type(self).__name__
        return '\n'.join(["{:s}[{:.5g} {:.5g} {:.5g}; {:.5g} {:.5g} {:.5g}]".format(typ, *list(x)) for x in self.data])

    def __neg__(self):
        """
        Unary minus for spatial vector

        ``-V`` is a spatial vector of the same type as ``V`` whose value is
        the negative of ``V``.  If V is an array V (1xN) then the result
        is an array (1xN).

        See also SpatialVec6.minus, SpatialVec6.plus.
        """

        # for i=1:numel(obj)
        # y(i) = obj.new(-obj(i).vw);

        return  self.__class__([-x for x in self.data])


    def __add__(left, right):
        """
        Addition for spatial vectors

        V1 + V2 is a spatial vector of the same type as V1 and V2 whose value is
        the sum of V1 and V2.  If both are arrays of spatial vectors V1 (1xN) and
        V2 (1xN) the result is an array (1xN).

        See also SpatialVec6.minus.
            :param right:
            :return:
        """
        assert type(left) == type(right), 'can only add spatial vectors of same type'
        assert len(left) == len(right), 'can only add equal length arrays of spatial vectors'

        return left.__class__([x + y for x, y in zip(left.data, right.data)])

    def __sub__(left, right):
        """
        Subtraction Addition for spatial vectors

        :param right:
        :return:

        V1 - V2 is a spatial vector of the same type as V1 and V2 whose value is
        the difference of V1 and V2.  If both are arrays of spatial vectors V1 (1xN) and
        V2 (1xN) the result is an array (1xN).

        See also SpatialVec6.__minus__, SpatialVec6.__add__
        """
        assert type(left) == type(right), 'can only subtract spatial vectors of same type'
        assert len(left) == len(right), 'can only subtract equal length arrays of spatial vectors'

        return left.__class__([x - y for x, y in zip(left.data, right.data)])


class SpatialM6(SpatialVector):
    """
    Create a new spatial motion class (abstract class)
    
    Abstract superclass that represents spatial motion.  This class has two
    concrete subclasses:

    Methods::
     SpatialM6     ^constructor invoked by subclasses
     char          ^convert to string
     cross         cross product
     display       ^display in human readable form
     double        ^convert to a 6xN double
    
    Operators::
     +          ^add spatial vectors of the same type
     -          ^subtract spatial vectors of the same type
     -          ^unary minus of spatial vectors
    
    Notes:
     - ^ is inherited from SpatialVec6.
     - Subclass of the MATLAB handle class which means that pass by reference semantics
       apply.
     - Spatial vectors can be placed into arrays and indexed.
    
    See also SpatialForce, SpatialMomentum, SpatialInertia, SpatialM6.
    """

    @abstractmethod
    def __init__(self, value):
        """
        Create a new spatial motion vector (abstract class)

        :param value:

        SpatiaVecXXX(V) is a spatial vector of type SpatiaVecXXX with a value
        from V (6x1).  If V (6xN) then an (Nx1) array of spatial vectors is
        returned.

        See also SpatialVelocity, SpatialAcceleration, SpatialForce, SpatialMomentum.
        """
        super().__init__(value)

    def cross(self, other):
        """

        :param right:
        :return:
        SpatialM6.cross Spatial velocity cross product

        - ``cross(V1, V2)`` is a SpatialAcceleration object where V1 and V2 are SpatialM6
        subclass instances.

        cross(V, F) is a SpatialForce object where V1 is a SpatialM6
        subclass instances and F is a SpatialForce subclass instance.

        Notes:

         - The first form is Featherstone's "x" operator.
         - The second form is Featherstone's "x*" operator.

        """
        pass

        # v = obj.vw;
        # # vcross = [ skew(w) skew(v); zeros(3,3) skew(w) ]
        
        v = self.V
        vcross = np.array([
                            [0,    -v[5],  v[5],   0,   -v[2],   v[1]],
                            [v[5],  0,    -v[3],   v[2],  0,    -v[0]],
                            [-v[4], v[3],  0,     -v[1],  v[0],  0],
                            [0,     0,     0,      0,    -v[5],  v[4]],
                            [0,     0,     0,      v[5],  0,    -v[3]],
                            [0,     0,     0,     -v[4],  v[3],  0]
                        ])
        if isinstance(other, SpatialVelocity):
            return SpatialAcceleration(vcross * other.V)  # * operator
        elif isinstance(other, SpatialF6):
            return SpatialAcceleration(-vcross * other.V)  # x* operator
        else:
            raise TypeError('type mismatch')

    def __mul(left, right):
        return left.cross(right)

    def __rmul(right, left):
        if isinstance(left, SpatialInertia):
            # result is SpatialMomentum
            pass  # TODO
        elif isinstance(left, Twist):
            # result is transformed SpatialVelocity or SpatialAcceleration
            # Twist * SpatialVelocity -> SpatialVelocity
            # Twist * SpatialAcceleration -> SpatialAcceleration
            return right.__class__(left.Ad.T @ right.V)
        else:
            raise ValueError('SpatialM6 with unknown premultiplication type')


class SpatialF6(SpatialVector):
    """
        Abstract spatial force class

        Abstract superclass that represents spatial force.  This class has two
        concrete subclasses:

        Operators:

         +          ^add spatial vectors of the same type
         -          ^subtract spatial vectors of the same type
         -          ^unary minus of spatial vectors

        Notes:

        - ^ is inherited from SpatialVec6.
        - Spatial vectors can be placed into arrays and indexed.

        See also SpatialForce, SpatialMomentum, SpatialInertia, SpatialM6.
        """

    @abstractmethod
    def __init__(self, value):
        super().__init__(value)


class SpatialVelocity(SpatialM6):
    """
    Spatial velocity class

    Concrete subclass of SpatialM6 that represents the
    translational and rotational velocity of a rigid-body moving in 3D space.

    Operators:

     +      ^add spatial vectors of the same type
     -      ^subtract spatial vectors of the same type
     -      ^unary minus of spatial vectors
     *      ^^^premultiplication by SpatialInertia yields SpatialMomentum
     *      ^^^^premultiplication by Twist yields transformed SpatialVelocity

    Notes:
    - ^ is inherited from SpatialVec6.
    - ^^ is inherited from SpatialM6.
    - ^^^ are implemented in SpatialInertia.
    - ^^^^ are implemented in Twist.

    See also SpatialVec6, SpatialM6, SpatialAcceleration, SpatialInertia, SpatialMomentum.


    """
    def __init__(self, value=None):
        super().__init__(value)

class SpatialAcceleration(SpatialM6):
    """
    Spatial acceleration class

    Concrete subclass of SpatialM6 that represents the
    translational and rotational acceleration of a rigid-body moving in 3D space.

    Methods:

     SpatialAcceleration    ^constructor invoked by subclasses
     char                   ^convert to string
     cross                  ^^cross product
     display                ^display in human readable form
     double                 ^convert to a 6xN double
     new                    construct new concrete class of same type

    Operators::
     +     ^add spatial vectors of the same type
     -     ^subtract spatial vectors of the same type
     -     ^unary minus of spatial vectors
     *     ^^^premultiplication by SpatialInertia yields SpatialForce
     *     ^^^^premultiplication by Twist yields transformed SpatialAcceleration


    Notes:
     - ^ is inherited from SpatialVec6.
     - ^^ is inherited from SpatialM6.
     - ^^^ are implemented in SpatialInertia.
     - ^^^^ are implemented in Twist.

        """
    def __init__(self, value=None):
        super().__init__(value)



class SpatialForce(SpatialF6):
    """
    Spatial force class

    Concrete subclass of SpatialF6 and represents the
    translational and rotational forces and torques acting on a rigid-body in 3D space.

    Operators::
     +          ^add spatial vectors of the same type
     -          ^subtract spatial vectors of the same type
     -          ^unary minus of spatial vectors
     *          ^^^premultiplication by SE3 yields transformed SpatialForce
     *          ^^^^premultiplication by Twist yields transformed SpatialForce

    Notes:
    - ^ is inherited from SpatialVec6.
    - ^^ is inherited from SpatialM6.
    - ^^^ are implemented in RTBPose.
    - ^^^^ are implemented in Twist.

    See also SpatialVec6, SpatialF6, SpatialMomentum.
    """
    
    def __init__(self, value=None):
        super().__init__(value)
# n = SpatialForce(val);

    def __rmul(right, left):
        # Twist * SpatialForce -> SpatialForce
        return SpatialForce(left.Ad.T @ right.V)


class SpatialMomentum(SpatialF6):

    """
    Spatial momentum class

    Operators::
     +          ^add spatial vectors of the same type
     -          ^subtract spatial vectors of the same type
     -          ^unary minus of spatial vectors

    Notes:
     - ^ is inherited from SpatialVec6.
     - ^^ is inherited from SpatialM6.

    See also SpatialVec6, SpatialF6, SpatialForce.
    """
    def __init__(self, value=None):
        super().__init__(value)

class SpatialInertia(UserList):
    """
    Spatial inertia class

    Concrete class representing spatial inertia.

    Methods:
     SpatialInertia   constructor
     char             convert to string
     display          display in human readable form
     double           convert to a 6xN double

    Operators:
     +          plus: add spatial inertia of connected bodies
     *          mtimes: compute force or momentum
    Notes:

     - Subclass of the MATLAB handle class which means that pass by reference semantics
       apply.
     - Spatial inertias can be placed into arrays and indexed.

    See also SpatialM6, SpatialF6, SpatialVelocity, SpatialAcceleration, SpatialForce,
    SpatialMomentum.
    """
    def __init__(self, m=None, c=None, I=None):
        """
        Create a new spatial inertia

        :param m: mass
        :type m: float
        :param c: centre of mass relative to link frame
        :type c: 3-element array_like
        :param I: inertia about the centre of mass, axes aligned with link frame
        :type I: numpy.array, shape=(6,6)

        - ``SpatialInertia(M, C, I)`` is a spatial inertia object for a rigid-body
          with mass ``M``, centre of mass at ``C`` relative to the link frame, and an
          inertia matrix ``I`` (3x3) about the centre of mass.

        - ``SpatialInertia(I)`` is a spatial inertia object with a value equal
          to ``I`` (6x6).
        """
        if m is not None and c is not None:
            assert arg.isvector(c, 3), 'c must be 3-vector'
            if I is None:
                I = np.zeros((3,3))
            else:
                assert arg.ismatrix(I, (3,3)), 'I must be 3x3 matrix'
            C = tr.skew(c)
            self.I = np.array([
                                [m * np.eye(3), m @ C.T],
                                [m @ C,         I + m * C * C.T]
                              ])
        elif m is None and c is None and I is not None:
            assert arg.ismatrix(I, (6, 6)), 'I must be 6x6 matrix'

    def __repr__(self):

        """
        Convert to string

        s = SI.char() is a string showing spatial inertia parameters in a
        compact format.
        If SI is an array of spatial inertia objects return a string with the
        inertia values in a vertical list.

        See also SpatialInertia.display.
        """
        return self.__str__()

    def __str__(self):
        return str(self.I)


    def __add__(left, right):
        """
        Spatial inertia addition
        :param left:
        :param right:
        :return:

        - ``SI1 + SI2`` is the SpatialInertia of a composite body when bodies with
           SpatialInertia ``SI1`` and ``SI2`` are connected.
        """

        assert type(left) == type(right), 'spatial inertia can only be added to spatial inertia'
        return SpatialInertia(a.I + b.I)

    def __mul__(left, right):
        """
        Spatial inertia product

        :param left:
        :param right:
        :return:

        - ``SI * A`` is the SpatialForce required for a body with SpatialInertia ``SI`` to accelerate with
          the SpatialAcceleration ``A``.
        - ``SI * V`` is the SpatialMomemtum of a body with SpatialInertia ``SI`` and SpatialVelocity ``V``.
        """

        if instance(right, SpatialAcceleration):
            v = SpatialForce(a.I * b.vw);  # F = ma
        elif instance(right, SpatialVelocity):
            # crf(v(i).vw)*model.I(i).I*v(i).vw;
            # v = Wrench( a.cross() * I.I * a.vw );
            v = SpatialMomentum(a.I * b.vw)   # M = mv
        else:
            raise TypeError('bad postmultiply operands for Inertia *')

    def __rmul__(right, left):
        """
        Spatial inertia product

        :param left:
        :param right:
        :return:

        - ``A * SI`` is the SpatialForce required for a body with SpatialInertia ``SI`` to accelerate with
          the SpatialAcceleration ``A``.
        - ``V * SI `` is the SpatialMomemtum of a body with SpatialInertia ``SI`` and SpatialVelocity ``V``.
        """
        return right.__mul__(left)

if __name__ == "__main__":

    import numpy.testing as nt
    import matplotlib.pyplot as plt
    import unittest

    class TestSpatialVector(unittest.TestCase):

        def test_list_powers(self):
            x = SpatialVelocity.Empty()
            self.assertEqual(len(x), 0)
            x.append(SpatialVelocity([1, 2, 3, 4, 5, 6]))
            self.assertEqual(len(x), 1)

            x.append(SpatialVelocity([7, 8, 9, 10, 11, 12]))
            self.assertEqual(len(x), 2)

            y = x[0]
            self.assertIsInstance(y, SpatialVelocity)
            self.assertEqual(len(y), 1)
            self.assertTrue(all(y.V == np.r_[1, 2, 3, 4, 5, 6]))

            y = x[1]
            self.assertIsInstance(y, SpatialVelocity)
            self.assertEqual(len(y), 1)
            self.assertTrue(all(y.V == np.r_[7, 8, 9, 10, 11, 12]))

            x.insert(0, SpatialVelocity([20, 21, 22, 23, 24, 25]))

            y = x[0]
            self.assertIsInstance(y, SpatialVelocity)
            self.assertEqual(len(y), 1)
            self.assertTrue(all(y.V == np.r_[20, 21, 22, 23, 24, 25]))

            y = x[1]
            self.assertIsInstance(y, SpatialVelocity)
            self.assertEqual(len(y), 1)
            self.assertTrue(all(y.V == np.r_[1, 2, 3, 4, 5, 6]))

        def test_velocity(self):
            a = SpatialVelocity([1, 2, 3, 4, 5, 6])
            self.assertIsInstance(a, SpatialVelocity)
            self.assertIsInstance(a, SpatialVector)
            self.assertIsInstance(a, SpatialM6)
            self.assertEqual(len(a), 1)
            self.assertTrue(all(a.V == np.r_[1, 2, 3, 4, 5, 6]))

            a = SpatialVelocity(np.r_[1, 2, 3, 4, 5, 6])
            self.assertIsInstance(a, SpatialVelocity)
            self.assertIsInstance(a, SpatialVector)
            self.assertIsInstance(a, SpatialM6)
            self.assertEqual(len(a), 1)
            self.assertTrue(all(a.V == np.r_[1, 2, 3, 4, 5, 6]))

            s = str(a)
            self.assertIsInstance(s, str)
            self.assertEqual(s.count('\n'), 0)
            self.assertTrue(s.startswith('SpatialVelocity'))

            r = np.random.rand(6, 10)
            a = SpatialVelocity(r)
            self.assertIsInstance(a, SpatialVelocity)
            self.assertIsInstance(a, SpatialVector)
            self.assertIsInstance(a, SpatialM6)
            self.assertEqual(len(a), 10)

            b = a[3]
            self.assertIsInstance(b, SpatialVelocity)
            self.assertIsInstance(b, SpatialVector)
            self.assertIsInstance(b, SpatialM6)
            self.assertEqual(len(b), 1)
            self.assertTrue(all(b.V == r[:,3]))

            s = str(a)
            self.assertIsInstance(s, str)
            self.assertEqual(s.count('\n'), 9)

        def test_acceleration(self):
            a = SpatialAcceleration([1, 2, 3, 4, 5, 6])
            self.assertIsInstance(a, SpatialAcceleration)
            self.assertIsInstance(a, SpatialVector)
            self.assertIsInstance(a, SpatialM6)
            self.assertEqual(len(a), 1)
            self.assertTrue(all(a.V == np.r_[1, 2, 3, 4, 5, 6]))

            a = SpatialAcceleration(np.r_[1, 2, 3, 4, 5, 6])
            self.assertIsInstance(a, SpatialAcceleration)
            self.assertIsInstance(a, SpatialVector)
            self.assertIsInstance(a, SpatialM6)
            self.assertEqual(len(a), 1)
            self.assertTrue(all(a.V == np.r_[1, 2, 3, 4, 5, 6]))

            s = str(a)
            self.assertIsInstance(s, str)
            self.assertEqual(s.count('\n'), 0)
            self.assertTrue(s.startswith('SpatialAcceleration'))

            r = np.random.rand(6, 10)
            a = SpatialAcceleration(r)
            self.assertIsInstance(a, SpatialAcceleration)
            self.assertIsInstance(a, SpatialVector)
            self.assertIsInstance(a, SpatialM6)
            self.assertEqual(len(a), 10)

            b = a[3]
            self.assertIsInstance(b, SpatialAcceleration)
            self.assertIsInstance(b, SpatialVector)
            self.assertIsInstance(b, SpatialM6)
            self.assertEqual(len(b), 1)
            self.assertTrue(all(b.V == r[:,3]))

            s = str(a)
            self.assertIsInstance(s, str)


        def test_force(self):

            a = SpatialForce([1, 2, 3, 4, 5, 6])
            self.assertIsInstance(a, SpatialForce)
            self.assertIsInstance(a, SpatialVector)
            self.assertIsInstance(a, SpatialF6)
            self.assertEqual(len(a), 1)
            self.assertTrue(all(a.V == np.r_[1, 2, 3, 4, 5, 6]))

            a = SpatialForce(np.r_[1, 2, 3, 4, 5, 6])
            self.assertIsInstance(a, SpatialForce)
            self.assertIsInstance(a, SpatialVector)
            self.assertIsInstance(a, SpatialF6)
            self.assertEqual(len(a), 1)
            self.assertTrue(all(a.V == np.r_[1, 2, 3, 4, 5, 6]))

            s = str(a)
            self.assertIsInstance(s, str)
            self.assertEqual(s.count('\n'), 0)
            self.assertTrue(s.startswith('SpatialForce'))

            r = np.random.rand(6, 10)
            a = SpatialForce(r)
            self.assertIsInstance(a, SpatialForce)
            self.assertIsInstance(a, SpatialVector)
            self.assertIsInstance(a, SpatialF6)
            self.assertEqual(len(a), 10)

            b = a[3]
            self.assertIsInstance(b, SpatialForce)
            self.assertIsInstance(b, SpatialVector)
            self.assertIsInstance(b, SpatialF6)
            self.assertEqual(len(b), 1)
            self.assertTrue(all(b.V == r[:, 3]))

            s = str(a)
            self.assertIsInstance(s, str)

        def test_momentum(self):

            a = SpatialMomentum([1, 2, 3, 4, 5, 6])
            self.assertIsInstance(a, SpatialMomentum)
            self.assertIsInstance(a, SpatialVector)
            self.assertIsInstance(a, SpatialF6)
            self.assertEqual(len(a), 1)
            self.assertTrue(all(a.V == np.r_[1, 2, 3, 4, 5, 6]))

            a = SpatialMomentum(np.r_[1, 2, 3, 4, 5, 6])
            self.assertIsInstance(a, SpatialMomentum)
            self.assertIsInstance(a, SpatialVector)
            self.assertIsInstance(a, SpatialF6)
            self.assertEqual(len(a), 1)
            self.assertTrue(all(a.V == np.r_[1, 2, 3, 4, 5, 6]))

            s = str(a)
            self.assertIsInstance(s, str)
            self.assertEqual(s.count('\n'), 0)
            self.assertTrue(s.startswith('SpatialMomentum'))

            r = np.random.rand(6, 10)
            a = SpatialMomentum(r)
            self.assertIsInstance(a, SpatialMomentum)
            self.assertIsInstance(a, SpatialVector)
            self.assertIsInstance(a, SpatialF6)
            self.assertEqual(len(a), 10)

            b = a[3]
            self.assertIsInstance(b, SpatialMomentum)
            self.assertIsInstance(b, SpatialVector)
            self.assertIsInstance(b, SpatialF6)
            self.assertEqual(len(b), 1)
            self.assertTrue(all(b.V == r[:, 3]))

            s = str(a)
            self.assertIsInstance(s, str)


        def test_arith(self):

            # just test SpatialVelocity since all types derive from same superclass

            r1 = np.r_[1, 2, 3, 4, 5, 6]
            r2 = np.r_[7, 8, 9, 10, 11, 12]
            a1 = SpatialVelocity(r1)
            a2 = SpatialVelocity(r2)

            self.assertTrue(all((a1 + a2).V == r1 + r2))
            self.assertTrue(all((a1 - a2).V == r1 - r2))
            self.assertTrue(all((-a1).V == -r1))

        def test_inertia(self):
            # constructor
            # addition
            pass

        def test_products(self):
            # v x v = a  *, v x F6 = a
            # a x I, I x a
            # v x I, I x v
            # twist x v, twist x a, twist x F
            pass


    unittest.main(buffer=True)