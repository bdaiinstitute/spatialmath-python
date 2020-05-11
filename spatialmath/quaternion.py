# Author: Aditya Dua
# 28 January, 2018

from collections import UserList
import math
import numpy as np

import spatialmath.base as tr
import spatialmath.base.quaternions as quat
import spatialmath.base.argcheck as argcheck
import spatialmath.pose3d as p3d


#TODO
# angle
# vectorized RPY in and out

class Quaternion(UserList):
    """
    A quaternion is a compact method of representing a 3D rotation that has
    computational advantages including speed and numerical robustness.
    
    A quaternion has 2 parts, a scalar s, and a 3-vector v and is typically written:
        q = s <vx vy vz>
    """
    
    def __init__(self, s=None, v=None, check=True, norm=True):
        """        
        A zero quaternion is one for which M{s^2+vx^2+vy^2+vz^2 = 1}.
        A quaternion can be considered as a rotation about a vector in space where
        q = cos (theta/2) sin(theta/2) <vx vy vz>
        where <vx vy vz> is a unit vector.
        :param s: scalar
        :param v: vector
        """
        if s is None and v is None:
            self.data = [ np.array([0, 0, 0, 0]) ]
            
        elif argcheck.isscalar(s) and argcheck.isvector(v,3):
            self.data = [ np.r_[s, argcheck.getvector(v)] ]
            
        elif argcheck.isvector(s,4):
            self.data = [ argcheck.getvector(s) ]
            
        elif type(s) is list:
            if isinstance(s[0], np.ndarray):
                if check:
                    assert argcheck.isvectorlist(s,4), 'list must comprise 4-vectors'
                self.data = s
            elif isinstance(s[0], self.__class__):
                # possibly a list of objects of same type
                assert all( map( lambda x: isinstance(x, self.__class__), s) ), 'all elements of list must have same type'
                self.data = [x._A for x in s]
            else:
                raise ValueError('incorrect list')
        
        elif isinstance(s, np.ndarray) and s.shape[1] == 4:
            self.data = [x for x in s]
            
        elif isinstance(s, Quaternion):
            self.data = s.data
            
        else:
            raise ValueError('bad argument to Quaternion constructor')
            
    def append(self, x):
        print('in append method')
        if not type(self) == type(x):
            raise ValueError("cant append different type of pose object")
        if len(x) > 1:
            raise ValueError("cant append a pose sequence - use extend")
        super().append(x._A)
        
    @property
    def _A(self):
        # get the underlying numpy array
        if len(self.data) == 1:
            return self.data[0]
        else:
            return self.data

    def __getitem__(self, i):
        #print('getitem', i)
        #return self.__class__(self.data[i])
        return self.__class__(self.data[i])


    @property
    def s(q):
        """
        :arg q: input quaternion
        :type q: Quaternion, UnitQuaternion
        :return: real part of quaternion
        :rtype: float or numpy.ndarray
        
        - If the quaternion is of length one, a scalar float is returned.
        - If the quaternion is of length >1, a numpy array shape=(N,) is returned.
        """
        if len(q) == 1:
            return q._A[0]
        else:
            return np.array([q.s for q in q])

    @property
    def v(q):
        """
        :arg q: input quaternion
        :type q: Quaternion, UnitQuaternion
        :return: vector part of quaternion
        :rtype: numpy ndarray
        
        - If the quaternion is of length one, a numpy array shape=(3,) is returned.
        - If the quaternion is of length >1, a numpy array shape=(N,3) is returned.
        """
        if len(q) == 1:
            return q._A[1:4]
        else:
            return np.array([q.v for q in q])
    
    @property
    def vec(q):
        """
        :arg q: input quaternion
        :type q: Quaternion, UnitQuaternion
        :return: quaternion expressed as a vector
        :rtype: numpy ndarray
        
        - If the quaternion is of length one, a numpy array shape=(4,) is returned.
        - If the quaternion is of length >1, a numpy array shape=(N,4) is returned.
        """
        if len(q) == 1:
            return q._A
        else:
            return np.array([q._A for q in q])
    

    @classmethod
    def pure(cls, v):
        return cls(s=0, v=argcheck.getvector(v,3), norm=True)
    
    @property
    def conj(self):
        return self.__class__( [quat.conj(q._A) for q in self], norm=False)



    @property
    def norm(self):
        """Return the norm of this quaternion.
        Code retrieved from: https://github.com/petercorke/robotics-toolbox-python/blob/master/robot/Quaternion.py
        Original authors: Luis Fernando Lara Tobar and Peter Corke
        @rtype: number
        @return: the norm
        """
        if len(self) == 1:
            return quat.qnorm(self._A)
        else:
            return np.array([quat.qnorm(q._A) for q in self])

    @property
    def unit(self):
        """Return an equivalent unit quaternion
        Code retrieved from: https://github.com/petercorke/robotics-toolbox-python/blob/master/robot/Quaternion.py
        Original authors: Luis Fernando Lara Tobar and Peter Corke
        @rtype: quaternion
        @return: equivalent unit quaternion
        """
        return UnitQuaternion( [quat.unit(q._A) for q in self], norm=False)


    @property
    def matrix(self):
        return quat.matrix(self._A)
    
    #-------------------------------------------- arithmetic
    
    def inner(self, other):
        assert isinstance(other, Quaternion), 'operands to inner must be Quaternion subclass'
        return self._op2(other, lambda x, y: quat.inner(x, y), list1=False )
    
    def __eq__(self, other):
        assert type(self) == type(other), 'operands to == are of different types'
        return self._op2(other, lambda x, y: quat.isequal(x, y), list1=False )
    
    def __ne__(self, other):
        assert type(self) == type(other), 'operands to == are of different types'
        return self._op2(other, lambda x, y: not quat.isequal(x, y), list1=False )
        
    
    def __mul__(left, right):
        """
        multiply quaternion
        
        :arg left: left multiplicand
        :type left: Quaternion
        :arg right: right multiplicand
        :type left: Quaternion, UnitQuaternion, float
        :return: product
        :rtype: Quaternion
        :raises: ValueError
        
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

        """
        if isinstance(right, left.__class__):
            # quaternion * [unit]quaternion case
            return Quaternion( left._op2(right, lambda x, y: quat.qqmul(x, y) ) )

        elif argcheck.isscalar(right):
            # quaternion * scalar case
            #print('scalar * quat')
            return Quaternion([right*q._A for q in left])

        else:
            raise ValueError('operands to * are of different types')
            
        return left._op2(right, lambda x, y: x @ y )

    def __rmul__(right, left):
        """
        Pre-multiply quaternion
        
        :arg right: right multiplicand
        :type right: Quaternion, 
        :arg left: left multiplicand
        :type left: float
        :return: product
        :rtype: Quaternion
        :raises: ValueError
        
        Premultiplies a quaternion by a scalar. If the right operand is a list, 
        the result will be a list .
        
        Example::
            
            q = Quaternion()
            q = 2 * q
        
        :seealso: :func:`__mul__`
        """
        # scalar * quaternion case
        return Quaternion([left*q._A for q in right])
        
    def __imul__(left, right):
        """
        Multiply quaternion in place
        
        :arg left: left multiplicand
        :type left: Quaternion
        :arg right: right multiplicand
        :type right: Quaternion, UnitQuaternion, float
        :return: product
        :rtype: Quaternion
        :raises: ValueError

        Multiplies a quaternion in place. If the right operand is a list, 
        the result will be a list.
        
        Example::
            
            q = Quaternion()
            q *= 2
            
        :seealso: :func:`__mul__`        

        """
        return left.__mul__(right)
    
    def __pow__(self, n):
        assert n >= 0, 'n must be >= 0, cannot invert a Quaternion'
        return self.__class__([quat.pow(q._A, n) for q in self])
    
    def __ipow__(self, n):
        return self.__pow__(n)
                    

    def __truediv__(self, other):
        raise NotImplemented('Quaternion division not supported')
    

    def __add__(left, right):
        """
        add quaternions
        
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
        """
        # results is not in the group, return an array, not a class
        assert type(left) == type(right), 'operands to + are of different types'
        return Quaternion( left._op2(right, lambda x, y: x + y ) )

    def __sub__(left, right):
        """
        subtract quaternions
        
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
        """
        # results is not in the group, return an array, not a class
        # TODO allow class +/- a conformant array
        assert type(left) == type(right), 'operands to - are of different types'
        return Quaternion( left._op2(right, lambda x, y: x - y ) )
    
    
    def _op2(self, other, op, list1=True):
        
        if len(self) == 1:
            if len(other) == 1:
                if list1:
                    return [op(self._A, other._A)]
                else:
                    return op(self._A, other._A)
            else:
                #print('== 1xN')
                return [op(self._A, x._A) for x in other]
        else:
            if len(other) == 1:
                #print('== Nx1')
                return [op(x._A, other._A) for x in self]
            elif len(self) == len(other):
                #print('== NxN')
                return [op(x._A, y._A) for (x,y) in zip(self, other)]
            else:
                raise ValueError('length of lists to == must be same length')
                
                

    # def __truediv__(self, other):
    #     assert isinstance(other, Quaternion) or isinstance(other, int) or isinstance(other,
    #                                                                                  float), "Can be divided by a " \
    #                                                                                          "Quaternion, " \
    #                                                                                          "int or a float "
    #     qr = Quaternion()
    #     if type(other) is Quaternion:
    #         qr = self * other.inv()
    #     elif type(other) is int or type(other) is float:
    #         qr.s = self.s / other
    #         qr.v = self.v / other
    #     return qr

    # def __eq__(self, other):
    #     # assert type(other) is Quaternion
    #     try:
    #         np.testing.assert_almost_equal(self.s, other.s)
    #     except AssertionError:
    #         return False
    #     if not matrices_equal(self.v, other.v, decimal=7):
    #         return False
    #     return True

    # def __ne__(self, other):
    #     if self == other:
    #         return False
    #     else:
    #         return True

    def __repr__(self):
        s = ''
        for q in self:
            s += quat.qprint(q._A, file=None) + '\n'
        s.rstrip('\n')
        return s

    def __str__(self):
        return self.__repr__()


    
class UnitQuaternion(Quaternion):
    r"""
    A unit-quaternion is is a quaternion with unit length, that is
    :math:`s^2+v_x^2+v_y^2+v_z^2 = 1`.
    
    A unit-quaternion can be considered as a rotation :math:`\theta`about a 
    unit-vector in space :math:`v=[v_x, v_y, v_z]` where
    :math:`q = \cos \theta/2 \sin \theta/2 <v_x v_y v_z>`.
    """
    
    def __init__(self, s=None, v=None, norm=True, check=True):
        """
        Construct a UnitQuaternion object
        
        :arg norm: explicitly normalize the quaternion [default True]
        :type norm: bool
        :arg check: explicitly check dimension of passed lists [default True]
        :type check: bool
        :return: new unit uaternion
        :rtype: UnitQuaternion
        :raises: ValueError
        
        Single element quaternion:
            
        - ``UnitQuaternion()`` constructs the identity quaternion 1<0,0,0>
        - ``UnitQuaternion(s, v)`` constructs a unit quaternion with specified
          real ``s`` and ``v`` vector parts. ``v`` is a 3-vector given as a 
          list, tuple, numpy.ndarray
        - ``UnitQuaternion(v)`` constructs a unit quaternion with specified 
          elements from ``v`` which is a 4-vector given as a list, tuple, numpy.ndarray
        - ``UnitQuaternion(R)`` constructs a unit quaternion from an orthonormal
          rotation matrix given as a 3x3 numpy.ndarray. If ``check`` is True
          test the matrix for orthogonality.
        
        Multi-element quaternion:
            
        - ``UnitQuaternion(V)`` constructs a unit quaternion list with specified 
          elements from ``V`` which is an Nx4 numpy.ndarray, each row is a
          quaternion.  If ``norm`` is True explicitly normalize each row.
        - ``UnitQuaternion(L)`` constructs a unit quaternion list from a list
          of 4-element numpy.ndarrays.  If ``check`` is True test each element
          of the list is a 4-vector. If ``norm`` is True explicitly normalize 
          each vector.
        """
        
        if s is None and v is None:
            self.data = [ quat.eye() ]
            
        elif argcheck.isscalar(s) and argcheck.isvector(v,3):
            q = np.r_[ s, argcheck.getvector(v) ]
            if norm:
                q = quat.unit(q)
            self.data = [q]
            
        elif argcheck.isvector(s,4):
            #print('uq constructor 4vec')
            q = argcheck.getvector(s)
            # if norm:
            #     q = quat.unit(q)
            #print(q)
            self.data = [quat.unit(s)]
            
        elif type(s) is list:
            if isinstance(s[0], np.ndarray):
                if check:
                    assert argcheck.isvectorlist(s,4), 'list must comprise 4-vectors'
                self.data = s
            elif isinstance(s[0], p3d.SO3):
                self.data = [quat.r2q(x.R) for x in s]
            
            elif isinstance(s[0], self.__class__):
                # possibly a list of objects of same type
                assert all( map( lambda x: type(x) == type(self), s) ), 'all elements of list must have same type'
                self.data = [x._A for x in s]
            else:
                raise ValueError('incorrect list')
        

        elif isinstance(s, p3d.SO3):
            self.data = [ quat.r2q(s.R) ]
            
        elif isinstance(s, np.ndarray) and tr.isrot(s, check=check):
            self.data = [ quat.r2q(s) ]
            
        elif isinstance(s, np.ndarray) and tr.ishom(s, check=check):
            self.data = [ quat.r2q(tr.t2r(s)) ]
            
        elif isinstance(s, np.ndarray) and s.shape[1] == 4:
            if norm:
                self.data = [quat.qnorm(x) for x in s]
            else:
                self.data = [x for x in s]
                
        elif isinstance(s, UnitQuaternion):
            self.data = s.data            
        else:
            raise ValueError('bad argument to UnitQuaternion constructor')

    # def __getitem__(self, i):
    #     print('uq getitem', i)
    #     #return self.__class__(self.data[i])
    #     return self.__class__(self.data[i])
    
    @property
    def R(self):
        return quat.q2r(self._A)
    
    @property
    def vec3(self):
        return quat.q2v(self._A)

    
    #-------------------------------------------- constructor variants
    @classmethod
    def Rx(cls, angle, unit='rad'):
        """
        Construct a UnitQuaternion object representing rotation about X-axis
        
        :arg angle: rotation angle
        :type norm: float
        :arg unit: rotation unit 'rad' [default] or 'deg'
        :type unit: str
        :return: new unit-quaternion
        :rtype: UnitQuaternion

        - ``UnitQuaternion(theta)`` constructs a unit quaternion representing a 
          rotation of `theta` radians about the X-axis.
        - ``UnitQuaternion(theta, 'deg')`` constructs a unit quaternion representing a 
          rotation of `theta` degrees about the X-axis.

        """
        return cls(tr.rotx(angle, unit=unit), check=False)

    @classmethod
    def Ry(cls, angle, unit='rad'):
        """
        Construct a UnitQuaternion object representing rotation about Y-axis
        
        :arg angle: rotation angle
        :type norm: float
        :arg unit: rotation unit 'rad' [default] or 'deg'
        :type unit: str
        :return: new unit-quaternion
        :rtype: UnitQuaternion

        - ``UnitQuaternion(theta)`` constructs a unit quaternion representing a 
          rotation of `theta` radians about the Y-axis.
        - ``UnitQuaternion(theta, 'deg')`` constructs a unit quaternion representing a 
          rotation of `theta` degrees about the Y-axis.

        """
        return cls(tr.roty(angle, unit=unit), check=False)

    @classmethod
    def Rz(cls, angle, unit='rad'):
        """
        Construct a UnitQuaternion object representing rotation about Z-axis
        
        :arg angle: rotation angle
        :type norm: float
        :arg unit: rotation unit 'rad' [default] or 'deg'
        :type unit: str
        :return: new unit-quaternion
        :rtype: UnitQuaternion

        - ``UnitQuaternion(theta)`` constructs a unit quaternion representing a 
          rotation of `theta` radians about the Z-axis.
        - ``UnitQuaternion(theta, 'deg')`` constructs a unit quaternion representing a 
          rotation of `theta` degrees about the Z-axis.

        """
        return cls(tr.rotz(angle, unit=unit), check=False)
    
    @classmethod
    def Rand(cls, N=1):
        """
        Create SO(3) with random rotation
    
        :param N: number of random rotations
        :type N: int
        :return: 3x3 rotation matrix
        :rtype: SO3 instance
    
        - ``SO3.Rand()`` is a random SO(3) rotation.
        - ``SO3.Rand(N)`` is an SO3 object containing a sequence of N random
          rotations.
        
        :seealso: :func:`spatialmath.quaternion.UnitQuaternion.Rand`
        """
        return cls( [quat.rand() for i in range(0,N)], check=False)
        

    @classmethod
    def Eul(cls, angles, *, unit='rad'):
        """
        Create an SO(3) rotation from Euler angles
    
        :param angles: 3-vector of Euler angles
        :type angles: array_like
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :return: 3x3 rotation matrix
        :rtype: SO3 instance
    
        ``SO3.Eul(ANGLES)`` is an SO(3) rotation defined by a 3-vector of Euler angles :math:`(\phi, \theta, \psi)` which
        correspond to consecutive rotations about the Z, Y, Z axes respectively.
          
        :seealso: :func:`~spatialmath.pose3d.SE3.eul`, :func:`~spatialmath.pose3d.SE3.Eul`, :func:`spatialmath.base.transforms3d.eul2r`
        """
        return cls(quat.r2q(tr.eul2r(angles, unit=unit)), check=False)

    @classmethod
    def RPY(cls, angles, *, order='zyx', unit='rad'):
        """
        Create an SO(3) rotation from roll-pitch-yaw angles
    
        :param angles: 3-vector of roll-pitch-yaw angles
        :type angles: array_like
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :param unit: rotation order: 'zyx' [default], 'xyz', or 'yxz'
        :type unit: str
        :return: 3x3 rotation matrix
        :rtype: SO3 instance
    
        ``SO3.RPY(ANGLES)`` is an SO(3) rotation defined by a 3-vector of roll, pitch, yaw angles :math:`(r, p, y)`
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
              
        :seealso: :func:`~spatialmath.pose3d.SE3.rpy`, :func:`~spatialmath.pose3d.SE3.RPY`, :func:`spatialmath.base.transforms3d.rpy2r`
        """
        return cls(quat.r2q(tr.rpy2r(angles, unit=unit, order=order)), check=False)

    @classmethod
    def OA(cls, o, a):
        """
        Create SO(3) rotation from two vectors
    
        :param o: 3-vector parallel to Y- axis
        :type o: array_like
        :param a: 3-vector parallel to the Z-axis
        :type o: array_like
        :return: 3x3 rotation matrix
        :rtype: SO3 instance
    
        ``SO3.OA(O, A)`` is an SO(3) rotation defined in terms of
        vectors parallel to the Y- and Z-axes of its reference frame.  In robotics these axes are 
        respectively called the orientation and approach vectors defined such that
        R = [N O A] and N = O x A.
    
        Notes:
            
        - The A vector is the only guaranteed to have the same direction in the resulting 
          rotation matrix
        - O and A do not have to be unit-length, they are normalized
        - O and A do not have to be orthogonal, so long as they are not parallel
        - The vectors O and A are parallel to the Y- and Z-axes of the equivalent coordinate frame.
    
        :seealso: :func:`spatialmath.base.transforms3d.oa2r`
        """
        return cls(quat.r2q(tr.oa2r(angles, unit=unit)), check=False)

    @classmethod
    def AngVec(cls, theta, v, *, unit='rad'):
        """
        Create an SO(3) rotation matrix from rotation angle and axis
    
        :param theta: rotation
        :type theta: float
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :param v: rotation axis, 3-vector
        :type v: array_like
        :return: 3x3 rotation matrix
        :rtype: SO3 instance
        
        ``SO3.AngVec(THETA, V)`` is an SO(3) rotation defined by
        a rotation of ``THETA`` about the vector ``V``.
        
        Notes:
            
        - If ``THETA == 0`` then return identity matrix.
        - If ``THETA ~= 0`` then ``V`` must have a finite length.
    
        :seealso: :func:`~spatialmath.pose3d.SE3.angvec`, :func:`spatialmath.base.transforms3d.angvec2r`
        """
        return cls(quat.r2q(tr.angvec2r(theta, v, unit=unit)), check=False)

    @classmethod
    def Omega(cls, w):

        return cls(quat.r2q(tr.angvec2r(tr.norm(w), tr.unitvec(w))), check=False)
    
    @classmethod
    def Vec3(cls, vec):
        return cls(quat.v2q(vec))
    

    @classmethod
    def angvec(cls, theta, v, unit='rad'):
        v = argcheck.getvector(v, 3)
        argcheck.isscalar(theta)
        theta = argcheck.getunit(theta, unit)
        return UnitQuaternion(s=math.cos(theta/2), v=math.sin(theta/2) * tr.unit(v), norm=False)

    def __truediv__(self, other):
        assert type(self) == type(other), 'operands to * are of different types'
        return self._op2(other, lambda x, y: quat.qqmul(x, quat.conj(y)) )
    
    @property
    def inv(self):
        return UnitQuaternion([quat.conj(q._A) for q in self])
    
    @classmethod
    def omega(cls, w):
        assert isvec(w, 3)
        theta = np.linalg.norm(w)
        s = math.cos(theta / 2)
        v = math.sin(theta / 2) * unitize(w)
        return cls(s=s, v=v)

    @staticmethod
    def qvmul(qv1, qv2):
        return quat.vvmul(qv1, qv2)
    
    def dot(self, omega):
        return tr.dot(self._A, omega)

    def dotb(self, omega):
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
        A 3-vector of length N is a 3xN numpy array, where each column is a 3-vector.
        
        :seealso: :func:`~spatialmath.Quaternion.__mul__`
        """
        if isinstance(left, right.__class__):
            # quaternion * quaternion case (same class)
            return right.__class__( left._op2(right, lambda x, y: quat.qqmul(x, y) ) )

        elif argcheck.isscalar(right):
            # quaternion * scalar case
            #print('scalar * quat')
            return Quaternion([right*q._A for q in left])
        
        
        elif isinstance(right, (list, tuple, np.ndarray)):
            #print('*: pose x array')
            if argcheck.isvector(right, 3):
                v = argcheck.getvector(right)
                if len(left) == 1:
                    # pose x vector
                    #print('*: pose x vector')
                    return quat.qvmul(left._A, argcheck.getvector(right,3))
                    
                elif len(left) > 1 and argcheck.isvector(right, 3):
                    # pose array x vector
                    #print('*: pose array x vector')
                    return np.array([tr.qvmul(x, v) for x in left._A]).T
                
            elif len(left) == 1 and isinstance(right, np.ndarray) and right.shape[0] == 3:
                return np.array([tr.qvmul(left._A, x) for x in right.T]).T
            else:
                raise ValueError('bad operands')
        else:
            raise ValueError('UnitQuaternion: operands to * are of different types')
            
        return left._op2(right, lambda x, y: x @ y )        
        


        return right.__mul__(left)
        
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
            
            q = UnitQuaternion()
            q *= 2
            
        :seealso: :func:`__mul__`        

        """
        return left.__mul__(right)


    def __truediv__(left, right):
        assert type(left) == type(right), 'operands to / are of different types'
        return UnitQuaternion( left._op2(right, lambda x, y: tr.qqmul(x, tr.conj(y)) ) )
    
    def __pow__(self, n):
        return self.__class__([quat.pow(q._A, n) for q in self])
    
    def __eq__(left, right):
        return left._op2(right, lambda x, y: quat.isequal(x, y, unitq=True), list1=False )
    
    def __ne__(left, right):
        return left._op2(right, lambda x, y: not quat.isequal(x, y, unitq=True), list1=False )
    
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
            assert type(dest) is UnitQuaternion
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


    def __repr__(self):
        s = ''
        for q in self:
            s += quat.qprint(q._A, delim=('<<', '>>'), file=None) + '\n'
        s.rstrip('\n')
        return s
    
    
    def __str__(self):
        return self.__repr__()

    def plot(self, *args, **kwargs):
        tr.trplot(tr.q2r(self._A), *args, **kwargs)
            
    @property
    def rpy(self, unit='rad', order='zyx'):
        return tr.tr2rpy(self.R, unit=unit, order=order)
    
    @property
    def eul(self, unit='rad', order='zyx'):
        return tr.tr2eul(self.R, unit=unit)

    @property
    def angvec(self, unit='rad'):
        return tr.tr2angvec(self.R)

    @property
    def SO3(self):
        return p3d.SO3(self.R, check=False)

    @property
    def SE3(self):
        return p3d.SE3(tr.r2t(self.R), check=False)


if __name__ == '__main__':  # pragma: no cover

    import pathlib
    import os.path
    
    exec(open(os.path.join(pathlib.Path(__file__).parent.absolute(), "test_quaternion.py")).read() )
