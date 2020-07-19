#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classes to represent orientation and pose in 3D.

@author: Peter Corke
"""

from collections import UserList
import numpy as np
import math

from spatialmath.base import argcheck
import spatialmath.base as tr
from spatialmath import super_pose as sp

# ============================== SO3 =====================================#

class SO3(sp.SMPose):
    """
    SO(3) subclass

    This subclass represents rotations in 3D space.  Internally it is a 3x3 orthogonal matrix belonging
    to the group SO(3) which describe rotations in 3D.

    .. inheritance-diagram::
    """

    def __init__(self, arg=None, *, check=True):
        """
        Construct new SO(3) object

        - ``SO3()`` is an SO3 instance representing null rotation -- the identity matrix
        - ``SO3(R)`` is an SO3 instance with rotation matrix R which is a 3x3 numpy array representing an valid rotation matrix.  If ``check``
          is ``True`` check the matrix value.
        - ``SO3([R1, R2, ... RN])`` where each Ri is a 3x3 numpy array of rotation matrices, is
          an SO3 instance containing N rotations. If ``check`` is ``True``
          then each matrix is checked for validity.
        - ``SO3([R1, R2, ... RN])`` where each Ri is an SO3 instance, is an SO3 instance containing N rotations.

        :seealso: `SMPose.pose_arghandler`
        """
        super().__init__()  # activate the UserList semantics

        if arg is None:
            # empty constructor
            if type(self) is SO3:
                self.data = [np.eye(3)]  # identity rotation
        else:
            super()._arghandler(arg, check=check)

    @property
    def R(self):
        """
        Rotational component as a matrix

        :param self: SO(3), SE(3)
        :type self: SO3 or SE3 instance
        :return: rotational component
        :rtype: numpy.ndarray

        ``T.R`` returns an:

        - ndarray with shape=(3,3), if len(T) == 1
        - ndarray with shape=(N,3,3), if len(T) = N > 1
        """
        if len(self) == 1:
            return self.A[:3, :3]
        else:
            return np.array([x[:3, :3] for x in self.A])

    @property
    def inv(self):
        """
        Inverse of SO(3)

        :param self: pose
        :type self: SE3 instance
        :return: inverse
        :rtype: SO2

        Returns the inverse, which for elements of SO(3) is the transpose.
        """
        if len(self) == 1:
            return SO3(self.A.T)
        else:
            return SO3([x.T for x in self.A])

    @property
    def n(self):
        """
        Normal vector of SO(3) pose

        :param self: pose
        :type self: SO3 instance
        :return: normal vector
        :rtype: numpy.ndarray, shape=(3,)

        Is the first column of the rotation submatrix, sometimes called the normal
        vector.  Parallel to the x-axis of the frame defined by this pose.
        """
        return self.A[:3, 0]

    @property
    def o(self):
        """
        Orientation vector of SO(3) pose

        :param self: pose
        :type self: SO3 instance
        :return: orientation vector
        :rtype: numpy.ndarray, shape=(3,)

        Is the second column of the rotation submatrix, sometimes called the orientation
        vector.  Parallel to the y-axis of the frame defined by this pose.
        """
        return self.A[:3, 1]

    @property
    def a(self):
        """
        Approach vector of SO(3) pose

        :param self: pose
        :type self: SO3 instance
        :return: approach vector
        :rtype: numpy.ndarray, shape=(3,)

        Is the third column of the rotation submatrix, sometimes called the approach
        vector.  Parallel to the z-axis of the frame defined by this pose.
        """
        return self.A[:3, 2]

    @property
    def eul(self, unit='deg'):
        """
        Extract Euler angles from SO(3) rotation

        :param self: rotation or pose
        :type angles: SO3, SE3 instance
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :return: 3-vector of Euler angles
        :rtype: numpy.ndarray, shape=(3,)

        ``R.eul`` is the Euler angle representation of the rotation.  Euler angles are
        a 3-vector :math:`(\phi, \theta, \psi)` which correspond to consecutive
        rotations about the Z, Y, Z axes respectively.

        ``R.eul`` returns an:

        - ndarray with shape=(3,), if len(R) == 1
        - ndarray with shape=(N,3), if len(R) = N > 1

        :seealso: :func:`~spatialmath.pose3d.SE3.Eul`, ::func:`spatialmath.base.transforms3d.tr2eul`
        """
        if len(self) == 1:
            return tr.tr2eul(self.A, unit=unit)
        else:
            return np.array([tr.tr2eul(x, unit=unit) for x in self.A]).T

    @property
    def rpy(self, unit='deg', order='zyx'):
        """
        Extract roll-pitch-yaw angles from SO(3) rotation

        :param self: rotation or pose
        :type angles: SO3, SE3 instance
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :return: 3-vector of roll-pitch-yaw angles
        :rtype: numpy.ndarray, shape=(3,)

        ``R.rpy`` is the roll-pitch-yaw angle representation of the rotation.  The angles are
        a 3-vector :math:`(r, p, y)` which correspond to successive rotations about the axes
        specified by ``order``:

            - 'zyx' [default], rotate by yaw about the z-axis, then by pitch about the new y-axis,
              then by roll about the new x-axis.  Convention for a mobile robot with x-axis forward
              and y-axis sideways.
            - 'xyz', rotate by yaw about the x-axis, then by pitch about the new y-axis,
              then by roll about the new z-axis. Covention for a robot gripper with z-axis forward
              and y-axis between the gripper fingers.
            - 'yxz', rotate by yaw about the y-axis, then by pitch about the new x-axis,
              then by roll about the new z-axis. Convention for a camera with z-axis parallel
              to the optic axis and x-axis parallel to the pixel rows.

        ``R.rpy`` returns an:

        - ndarray with shape=(3,), if len(R) == 1
        - ndarray with shape=(N,3), if len(R) = N > 1

        :seealso: :func:`~spatialmath.pose3d.SE3.RPY`, ::func:`spatialmath.base.transforms3d.tr2rpy`
        """
        if len(self) == 1:
            return tr.tr2rpy(self.A, unit=unit)
        else:
            return np.array([tr.tr2rpy(x, unit=unit) for x in self.A]).T

    @property
    def Ad(self):
        """
        :param self: pose
        :type self: SE3 instance
        :return: adjoint matrix
        :rtype: numpy.ndarray, shape=(6,6)
        
        SE3.Ad  Adjoint matrix

        A = P.Ad() is the adjoint matrix (6x6) corresponding to the pose P.
        See also Twist.ad.

        """

        return np.r_[ np.c_[self.R, tr.skew(self.t) @ self.R],
                      np.c_[np.zeros((3,3)), self.R]
                        ]

    @staticmethod
    def isvalid(x):
        """
        Test if matrix is valid SO(3)

        :param x: matrix to test
        :type x: numpy.ndarray
        :return: true of the matrix is 3x3 and a valid element of SO(3), ie. it is an
            orthonormal matrix with determinant of +1.
        :rtype: bool

        :seealso: :func:`~spatialmath.base.transform3d.isrot`
        """
        return tr.isrot(x, check=True)

    @classmethod
    def Rx(cls, theta, unit='rad'):
        """
        Create SO(3) rotation about X-axis

        :param theta: rotation angle about the X-axis
        :type theta: float or array_like
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :return: 3x3 rotation matrix
        :rtype: SO3 instance

        - ``SE3.Rx(THETA)`` is an SO(3) rotation of THETA radians about the x-axis
        - ``SE3.Rx(THETA, "deg")`` as above but THETA is in degrees
        
        If ``theta`` is an array then the result is a sequence of rotations defined by consecutive
        elements.
        """
        return cls([tr.rotx(x, unit=unit) for x in argcheck.getvector(theta)], check=False)

    @classmethod
    def Ry(cls, theta, unit='rad'):
        """
        Create SO(3) rotation about the Y-axis

        :param theta: rotation angle about Y-axis
        :type theta: float or array_like
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :return: 3x3 rotation matrix
        :rtype: SO3 instance

        - ``SO3.Ry(THETA)`` is an SO(3) rotation of THETA radians about the y-axis
        - ``SO3.Ry(THETA, "deg")`` as above but THETA is in degrees
        
        If ``theta`` is an array then the result is a sequence of rotations defined by consecutive
        elements.
        """
        return cls([tr.roty(x, unit=unit) for x in argcheck.getvector(theta)], check=False)

    @classmethod
    def Rz(cls, theta, unit='rad'):
        """
        Create SO(3) rotation about the Z-axis

        :param theta: rotation angle about Z-axis
        :type theta: float or array_like
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :return: 3x3 rotation matrix
        :rtype: SO3 instance
        
        - ``SO3.Rz(THETA)`` is an SO(3) rotation of THETA radians about the z-axis
        - ``SO3.Rz(THETA, "deg")`` as above but THETA is in degrees
        
        If ``theta`` is an array then the result is a sequence of rotations defined by consecutive
        elements.
        """
        return cls([tr.rotz(x, unit=unit) for x in argcheck.getvector(theta)], check=False)

    @classmethod
    def Rand(cls, N=1):
        """
        Create SO(3) with random rotation

        :param N: number of random rotations
        :type N: int
        :return: SO(3) rotation matrix
        :rtype: SO3 instance

        - ``SO3.Rand()`` is a random SO(3) rotation.
        - ``SO3.Rand(N)`` is a sequence of N random rotations.

        :seealso: :func:`spatialmath.quaternion.UnitQuaternion.Rand`
        """
        return cls([tr.q2r(tr.rand()) for i in range(0, N)], check=False)

    @classmethod
    def Eul(cls, angles, *, unit='rad'):
        """
        Create an SO(3) rotation from Euler angles

        :param angles: 3-vector of Euler angles
        :type angles: array_like or numpy.ndarray with shape=(N,3)
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :return: 3x3 rotation matrix
        :rtype: SO3 instance

        ``SO3.Eul(angles)`` is an SO(3) rotation defined by a 3-vector of Euler angles :math:`(\phi, \theta, \psi)` which
        correspond to consecutive rotations about the Z, Y, Z axes respectively.
        
        If ``angles`` is an Nx3 matrix then the result is a sequence of rotations each defined by Euler angles
        correponding to the rows of angles.

        :seealso: :func:`~spatialmath.pose3d.SE3.eul`, :func:`~spatialmath.pose3d.SE3.Eul`, :func:`spatialmath.base.transforms3d.eul2r`
        """
        if argcheck.isvector(angles, 3):
            return cls(tr.eul2r(angles, unit=unit))
        else:
            return cls([tr.eul2r(a, unit=unit) for a in angles])

    @classmethod
    def RPY(cls, angles, *, order='zyx', unit='rad'):
        """
        Create an SO(3) rotation from roll-pitch-yaw angles

        :param angles: 3-vector of roll-pitch-yaw angles
        :type angles: array_like or numpy.ndarray with shape=(N,3)
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :param unit: rotation order: 'zyx' [default], 'xyz', or 'yxz'
        :type unit: str
        :return: 3x3 rotation matrix
        :rtype: SO3 instance

        ``SO3.RPY(angles)`` is an SO(3) rotation defined by a 3-vector of roll, pitch, yaw angles :math:`(r, p, y)`
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

        If ``angles`` is an Nx3 matrix then the result is a sequence of rotations each defined by RPY angles
        correponding to the rows of angles.
        
        :seealso: :func:`~spatialmath.pose3d.SE3.rpy`, :func:`~spatialmath.pose3d.SE3.RPY`, :func:`spatialmath.base.transforms3d.rpy2r`
        """
        if argcheck.isvector(angles, 3):
                return cls(tr.rpy2r(angles, order=order, unit=unit))
        else:
            return cls([tr.rpy2r(a, order=order, unit=unit) for a in angles])

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
        return cls(tr.oa2r(o, a), check=False)

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
        return cls(tr.angvec2r(theta, v, unit=unit), check=False)
    
    @classmethod
    def Exp(cls, S, so3=False):
        """
        Create an SO(3) rotation matrix from so(3)

        :param S: Lie algebra so(3)
        :type S: numpy ndarray
        :param so3: accept input as an so(3) matrix [default False]
        :type so3: bool
        :return: 3x3 rotation matrix
        :rtype: SO3 instance

        - ``SO3.Exp(S)`` is an SO(3) rotation defined by its Lie algebra
          which is a 3x3 so(3) matrix (skew symmetric)
        - ``SO3.Exp(t)`` is an SO(3) rotation defined by a 3-element twist
          vector (the unique elements of the so(3) skew-symmetric matrix)
        - ``SO3.Exp(T)`` is a sequence of SO(3) rotations defined by an Nx3 matrix
          of twist vectors, one per row.
          
        Note:
            
        - an input 3x3 matrix is ambiguous, it could be the first or third case above.  In this
          case the parameter `so3` is the decider.

        :seealso: :func:`spatialmath.base.transforms3d.trexp`, :func:`spatialmath.base.transformsNd.skew`
        """
        if argcheck.ismatrix(S, (-1,3)) and not so3:
            return cls([tr.trexp(s) for s in S])
        else:
            return cls(tr.trexp(S), check=False)

# ============================== SE3 =====================================#


class SE3(SO3):

    def __init__(self, x=None, y=None, z=None, *, check=True):
        """
        Construct new SE(3) object

        :param x: translation distance along the X-axis
        :type x: float
        :param y: translation distance along the Y-axis
        :type y: float
        :param z: translation distance along the Z-axis
        :type z: float
        :return: 4x4 homogeneous transformation matrix
        :rtype: SE3 instance
        
        - ``SE3()`` is a null motion -- the identity matrix
        - ``SE3(x, y, z)`` is a pure translation of (x,y,z)
        - ``SE3(T)`` where T is a 4x4 numpy array representing an SE(3) matrix.  If ``check``
          is ``True`` check the matrix belongs to SE(3).
        - ``SE3([T1, T2, ... TN])`` where each Ti is a 4x4 numpy array representing an SE(3) matrix, is
          an SE3 instance containing N rotations. If ``check`` is ``True``
          check the matrix belongs to SE(3).
        - ``SE3([T1, T2, ... TN])`` where each Ri is an SE3 instance, is an SE3 instance containing N rotations.
        """
        super().__init__()  # activate the UserList semantics

        if x is None:
            # SE3()
            # empty constructor
            self.data = [np.eye(4)]
        elif y is not None and z is not None:
                # SE3(x, y, z)
                self.data = [tr.transl(x, y, z)]
        elif y is None and z is None:
            if argcheck.isvector(x, 3):
                # SE3( [x, y, z] )
                self.data = [tr.transl(x)]
            elif isinstance(x, np.ndarray) and x.shape[1] == 3:
                # SE3( Nx3 )
                self.data = [tr.transl(T) for T in x]   
            else:
                super()._arghandler(x, check=check)
        else:
            raise ValueError('bad argument to constructor')

    @property
    def t(self):
        """
        Translational component of SE(3)

        :param self: SE(3)
        :type self: SE3 instance
        :return: translational component
        :rtype: numpy.ndarray

        ``T.t`` returns an:

        - ndarray with shape=(3,), if len(T) == 1
        - ndarray with shape=(N,3), if len(T) = N > 1
        """
        if len(self) == 1:
            return self.A[:3, 3]
        else:
            return np.array([x[:3, 3] for x in self.A])

    @property
    def T(self):
        raise NotImplemented('transpose is not meaningful for SE3 object')

    @property
    def inv(self):
        r"""
        Inverse of SE(3)

        :param self: pose
        :type self: SE3 instance
        :return: inverse
        :rtype: SE3

        Returns the inverse taking into account its structure

        :math:`T = \left[ \begin{array}{cc} R & t \\ 0 & 1 \end{array} \right], T^{-1} = \left[ \begin{array}{cc} R^T & -R^T t \\ 0 & 1 \end{array} \right]`
        """
        if len(self) == 1:
            return SE3(tr.trinv(self.A))
        else:
            return SE3([tr.trinv(x) for x in self.A])

    @staticmethod
    def isvalid(x):
        """
        Test if matrix is valid SE(3)

        :param x: matrix to test
        :type x: numpy.ndarray
        :return: true of the matrix is 4x4 and a valid element of SE(3), ie. it is an
            homogeneous transformation matrix.
        :rtype: bool

        :seealso: :func:`~spatialmath.base.transform3d.ishom`
        """
        return tr.ishom(x, check=True)

    @classmethod
    def Rx(cls, theta, unit='rad'):
        """
        Create SE(3) pure rotation about the X-axis

        :param theta: rotation angle about X-axis
        :type theta: float
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :return: 4x4 homogeneous transformation matrix
        :rtype: SE3 instance

        - ``SE3.Rx(THETA)`` is an SO(3) rotation of THETA radians about the x-axis
        - ``SE3.Rx(THETA, "deg")`` as above but THETA is in degrees
        
        If ``theta`` is an array then the result is a sequence of rotations defined by consecutive
        elements.
        """
        return cls([tr.trotx(x, unit) for x in argcheck.getvector(theta)])

    @classmethod
    def Ry(cls, theta, unit='rad'):
        """
        Create SE(3) pure rotation about the Y-axis

        :param theta: rotation angle about X-axis
        :type theta: float
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :return: 4x4 homogeneous transformation matrix
        :rtype: SE3 instance

        - ``SE3.Ry(THETA)`` is an SO(3) rotation of THETA radians about the y-axis
        - ``SE3.Ry(THETA, "deg")`` as above but THETA is in degrees

        If ``theta`` is an array then the result is a sequence of rotations defined by consecutive
        elements.
        """
        return cls([tr.troty(x, unit) for x in argcheck.getvector(theta)])

    @classmethod
    def Rz(cls, theta, unit='rad'):
        """
        Create SE(3) pure rotation about the Z-axis

        :param theta: rotation angle about Z-axis
        :type theta: float
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :return: 4x4 homogeneous transformation matrix
        :rtype: SE3 instance

        - ``SE3.Rz(THETA)`` is an SO(3) rotation of THETA radians about the z-axis
        - ``SE3.Rz(THETA, "deg")`` as above but THETA is in degrees
        
        If ``theta`` is an array then the result is a sequence of rotations defined by consecutive
        elements.
        """
        return cls([tr.trotz(x, unit) for x in argcheck.getvector(theta)])

    @classmethod
    def Rand(cls, *, xrange=[-1, 1], yrange=[-1, 1], zrange=[-1, 1], N=1):
        """
        Create a random SE(3)

        :param xrange: x-axis range [min,max], defaults to [-1, 1]
        :type xrange: 2-element sequence, optional
        :param yrange: y-axis range [min,max], defaults to [-1, 1]
        :type yrange: 2-element sequence, optional
        :param zrange: z-axis range [min,max], defaults to [-1, 1]
        :type zrange: 2-element sequence, optional
        :param N: number of random transforms
        :type N: int
        :return: homogeneous transformation matrix
        :rtype: SE3 instance
        
        Return an SE3 instance with random rotation and translation.

        - ``SE3.Rand()`` is a random SE(3) translation.
        - ``SE3.Rand(N)`` is an SE3 object containing a sequence of N random
          poses.

        :seealso: `~spatialmath.quaternion.UnitQuaternion.Rand`
        """
        X = np.random.uniform(low=xrange[0], high=xrange[1], size=N)  # random values in the range
        Y = np.random.uniform(low=yrange[0], high=yrange[1], size=N)  # random values in the range
        Z = np.random.uniform(low=yrange[0], high=zrange[1], size=N)  # random values in the range
        R = SO3.Rand(N=N)
        return cls([tr.transl(x, y, z) @ tr.r2t(r.A) for (x, y, z, r) in zip(X, Y, Z, R)])

    @classmethod
    def Eul(cls, angles, unit='rad'):
        """
        Create an SE(3) pure rotation from Euler angles

        :param angles: 3-vector of Euler angles
        :type angles: array_like or numpy.ndarray with shape=(N,3)
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :return: 4x4 homogeneous transformation matrix
        :rtype: SE3 instance

        ``SE3.Eul(ANGLES)`` is an SO(3) rotation defined by a 3-vector of Euler angles :math:`(\phi, \theta, \psi)` which
        correspond to consecutive rotations about the Z, Y, Z axes respectively.
        
        If ``angles`` is an Nx3 matrix then the result is a sequence of rotations each defined by Euler angles
        correponding to the rows of angles.

        :seealso: :func:`~spatialmath.pose3d.SE3.eul`, :func:`~spatialmath.pose3d.SE3.Eul`, :func:`spatialmath.base.transforms3d.eul2r`
        """
        if argcheck.isvector(angles, 3):
            return cls(tr.eul2tr(angles, unit=unit))
        else:
            return cls([tr.eul2tr(a, unit=unit) for a in angles])

    @classmethod
    def RPY(cls, angles, order='zyx', unit='rad'):
        """
        Create an SO(3) pure rotation from roll-pitch-yaw angles

        :param angles: 3-vector of roll-pitch-yaw angles
        :type angles: array_like or numpy.ndarray with shape=(N,3)
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :param unit: rotation order: 'zyx' [default], 'xyz', or 'yxz'
        :type unit: str
        :return: 4x4 homogeneous transformation matrix
        :rtype: SE3 instance

        ``SE3.RPY(ANGLES)`` is an SE(3) rotation defined by a 3-vector of roll, pitch, yaw angles :math:`(r, p, y)`
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
              
        If ``angles`` is an Nx3 matrix then the result is a sequence of rotations each defined by RPY angles
        correponding to the rows of angles.

        :seealso: :func:`~spatialmath.pose3d.SE3.rpy`, :func:`~spatialmath.pose3d.SE3.RPY`, :func:`spatialmath.base.transforms3d.rpy2r`
        """
        if argcheck.isvector(angles, 3):
                return cls(tr.rpy2tr(angles, order=order, unit=unit))
        else:
            return cls([tr.rpy2tr(a, order=order, unit=unit) for a in angles])

    @classmethod
    def OA(cls, o, a):
        """
        Create SE(3) pure rotation from two vectors

        :param o: 3-vector parallel to Y- axis
        :type o: array_like
        :param a: 3-vector parallel to the Z-axis
        :type o: array_like
        :return: 4x4 homogeneous transformation matrix
        :rtype: SE3 instance

        ``SE3.OA(O, A)`` is an SE(3) rotation defined in terms of
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
        return cls(tr.oa2tr(o, a))

    @classmethod
    def AngVec(cls, theta, v, *, unit='rad'):
        """
        Create an SE(3) pure rotation matrix from rotation angle and axis

        :param theta: rotation
        :type theta: float
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :param v: rotation axis, 3-vector
        :type v: array_like
        :return: 4x4 homogeneous transformation matrix
        :rtype: SE3 instance

        ``SE3.AngVec(THETA, V)`` is an SE(3) rotation defined by
        a rotation of ``THETA`` about the vector ``V``.

        Notes:

        - If ``THETA == 0`` then return identity matrix.
        - If ``THETA ~= 0`` then ``V`` must have a finite length.

        :seealso: :func:`~spatialmath.pose3d.SE3.angvec`, :func:`spatialmath.base.transforms3d.angvec2r`
        """
        return cls(tr.angvec2tr(theta, v, unit=unit))

    @classmethod
    def Exp(cls, S):
        """
        Create an SE(3) rotation matrix from se(3)

        :param S: Lie algebra se(3)
        :type S: numpy ndarray
        :return: 3x3 rotation matrix
        :rtype: SO3 instance

        - ``SE3.Exp(S)`` is an SE(3) rotation defined by its Lie algebra
          which is a 3x3 se(3) matrix (skew symmetric)
        - ``SE3.Exp(t)`` is an SE(3) rotation defined by a 6-element twist
          vector (the unique elements of the se(3) skew-symmetric matrix)

        :seealso: :func:`spatialmath.base.transforms3d.trexp`, :func:`spatialmath.base.transformsNd.skew`
        """
        if isinstance(S, np.ndarray) and S.shape[1] == 6:
            return cls([tr.trexp(s) for s in S])
        else:
            return cls(tr.trexp(S), check=False)
    
    @classmethod
    def Tx(cls, x):
        """
        Create SE(3) translation along the X-axis

        :param theta: translation distance along the X-axis
        :type theta: float
        :return: 4x4 homogeneous transformation matrix
        :rtype: SE3 instance

        `SE3.Tz(D)`` is an SE(3) translation of D along the x-axis
        """
        return cls(tr.transl(x, 0, 0))

    @classmethod
    def Ty(cls, y):
        """
        Create SE(3) translation along the Y-axis

        :param theta: translation distance along the Y-axis
        :type theta: float
        :return: 4x4 homogeneous transformation matrix
        :rtype: SE3 instance

        `SE3.Tz(D)`` is an SE(3) translation of D along the y-axis
        """
        return cls(tr.transl(0, y, 0))

    @classmethod
    def Tz(cls, z):
        """
        Create SE(3) translation along the Z-axis

        :param theta: translation distance along the Z-axis
        :type theta: float
        :return: 4x4 homogeneous transformation matrix
        :rtype: SE3 instance

        `SE3.Tz(D)`` is an SE(3) translation of D along the z-axis
        """
        return cls(tr.transl(0, 0, z))

# ============================== Twist =====================================#

class Twist(sp.SMTwist):
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

    # properties (SetAccess = protected)
    #     v  %axis direction (column vector)
    #     w  %moment (column vector)
    # end



    # ------------------------- constructors -------------------------------#

    def __init__(self, arg=None, w=None, check=True):
        """
        Twist.Twist Create Twist object

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
            self.data = [np.r_[0, 0, 0, 0, 0, 0]]
        
        elif isinstance(arg, Twist):
            # clone it
            self.data = [np.r_[arg.v, arg.w]]
            
        elif argcheck.isvector(arg, 6):
            s = argcheck.getvector(arg)
            self.data = [s]
            
        elif argcheck.isvector(arg, 3) and argcheck.isvector(w, 3):
            v = argcheck.getvector(arg)
            w = argcheck.getvector(w)
            self.data = [np.r_[v, w]]
            
        elif isinstance(arg, SE3):
            S = tr.trlog(arg.A)  # use closed form for SE(3)

            skw, v = tr.tr2rt(S)
            w = tr.vex(skw)
            self.data = [np.r_[v, w]]

        elif Twist.isvalid(arg):
            # it's an augmented skew matrix, unpack it
            skw, v = tr.tr2rt(arg)
            w = tr.vex(skw)
            self.data = [np.r_[v, w]]
            
        elif isinstance(arg, list):
            # construct from a list

            if isinstance(arg[0], np.ndarray):
                # possibly a list of numpy arrays
                if check:
                    assert all(map(lambda x: Twist.isvalid(x), arg)), 'all elements of list must have valid shape and value for the class'
                self.data = arg
            elif type(arg[0]) == type(self):
                # possibly a list of objects of same type
                assert all(map(lambda x: type(x) == type(self), arg)), 'all elements of list must have same type'
                self.data = [x.S for x in arg]
            elif type(arg[0]) == list:
                # possibly a list of 6-lists
                assert all(map(lambda x: isinstance(x, list) and len(x) == 6, arg)), 'all elements of list must have same type'
                self.data = [np.r_[x] for x in arg]
            else:
                raise ValueError('bad list argument to constructor')

        else:
            raise ValueError('bad argument to constructor')

    @classmethod
    def R(cls, a, q, p=None):
        
        w = tr.unitvec(argcheck.getvector(a, 3))
        v = -np.cross(w, argcheck.getvector(q, 3))
        if p is not None:
            pitch = argcheck.getvector(p, 3)
            v = v + pitch * w
        return cls(v, w)

    @classmethod
    def P(cls, a):
        w = np.r_[0, 0, 0]
        v = tr.unitvec(argcheck.getvector(a, 3))

        return cls(v, w)
    

    
    
    # ------------------------- static methods -------------------------------#

    @staticmethod
    def isvalid(v, check=True):
        if argcheck.isvector(v, 6):
            return True
        elif argcheck.ismatrix(v, (4,4)):
            # maybe be an se(3)
            if not all(v.diagonal() == 0):  # check diagonal is zero 
                return False
            if not all(v[3,:] == 0):  # check bottom row is zero
                return False
            if not tr.isskew(v[:3,:3]):
                  # top left 3x3 is skew symmetric
                  return False
            return True
        return False


    # ------------------------- properties -------------------------------#
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
    def S(self):
        """
        Twist vector

        TW.S is the twist vector in se(3) as a vector (6x1).

        Notes:

        - Sometimes referred to as the twist coordinate vector.
        """
        # get the underlying numpy array
        if len(self.data) == 1:
            return self.data[0]
        else:
            return self.data
    
    @property
    def v(self):
        return self.data[0][:3]
    
    @property
    def w(self):
        return self.data[0][3:6]

    @property
    def isprismatic(self):
        return tr.iszerovec(self.w)
    
    @property
    def ad(self):
        """
        Twist.ad Logarithm of adjoint

        TW.ad is the logarithm of the adjoint matrix of the corresponding
        homogeneous transformation.

        See also SE3.Ad.
        """
        x = np.array([skew(self.w), skew(self.v), [np.zeros((3,3)), skew(self.w)]])

    @property
    def Ad(self):
        """
        Twist.Ad Adjoint

        TW.Ad is the adjoint matrix of the corresponding
        homogeneous transformation.

        See also SE3.Ad.
        """

        return self.SE3.Ad
    
    @property
    def SE3(self):
        """
        Twist.SE Convert twist to SE2 or SE3 object

        TW.SE is an SE2 or SE3 object representing the homogeneous transformation equivalent to the twist.

        See also Twist.T, SE2, SE3.
        """

        return SE3(self.exp())

    @property
    def se3(self):
        """
        Twist.se Return the twist matrix

        TW.se is the twist matrix in se(2) or se(3) which is an augmented
        skew-symmetric matrix (3x3 or 4x4).

        """
        if len(self) == 1:
            return tr.skewa(self.S)
        else:
            return [tr.skewa(x.S) for x in self]
    
    @property
    def pitch(self):
        """
        %Twist.pitch Pitch of the twist
        %
        TW.pitch is the pitch of the Twist as a scalar in units of distance per radian.
            %
        Notes::
        - For 3D case only.
        """

        return np.dot(self.w, self.v)

    
    @property
    def line(self):
        """
        Twist.line Line of twist axis in Plucker form

        TW.line is a Plucker object representing the line of the twist axis.

        Notes:

        - For 3D case only.

        See also Plucker.
        """
        
        return Plucker([np.r_[tw.v - tw.pitch * tw.w, tw.w] for tw in self])


    @property
    def pole(self):
        """
        %Twist.pole Point on the twist axis
        %
        TW.pole is a point on the twist axis (2x1 or 3x1).
            %
        Notes::
        - For pure translation this point is at infinity.
        """

        return np.cross(self.w, self.v) / self.theta

    @property
    def theta(self):
        """
        Twist.theta Twist rotation

        TW.theta is the rotation (1x1) about the twist axis in radians.
        """

        return tr.norm(self.w)
    
    # ------------------------- methods -------------------------------#

    def __mul__(left, right):
        """
        Twist.mtimes Multiply twist by twist or scalar

        TW1 * TW2 is a new Twist representing the composition of twists TW1 and
        TW2.

        TW * T is an SE2 or SE3 that is the composition of the twist TW and the
        homogeneous transformation object T.

        TW * S with its twist coordinates scaled by scalar S.

        TW * T compounds a twist with an SE2/3 transformation
        %
        """
        
        if isinstance(right, Twist):
            # twist composition
            return Twist( left.exp() * right.exp());
        elif isinstance(right, (int, np.int64, float, np.float64)):
            return Twist(left.S * right)
        elif isinstance(right, SpatialVelocity):
            return SpatialVelocity(a.Ad @ b.vw)
        elif isinstance(right, SpatialAcceleration):
            return SpatialAcceleration(a.Ad @ b.vw)
        elif isinstance(right, SpatialForce):
            return SpatialForce(a.Ad @ b.vw)
        else:
            raise ValueError('twist *, incorrect right operand')

    def __imul__(left,right):
        return left.__mul__(right)

    def __rmul(right, left):
        if isinstance(left, (int, np.int64, float, np.float64)):
            return Twist(right.S * left)
        else:
            raise ValueError('twist *, incorrect left operand')
            
    def exp(self, theta=None, units='rad'):
        """
        Twist.exp Convert twist to homogeneous transformation

        TW.exp is the homogeneous transformation equivalent to the twist (SE2 or SE3).

        TW.exp(THETA) as above but with a rotation of THETA about the twist.

        Notes::
        - For the second form the twist must, if rotational, have a unit rotational component.

        See also Twist.T, trexp, trexp2.
        """
 
        if units != 'rad' and self.isprismatic:
            print('Twist.exp: using degree mode for a prismatic twist')


        if theta is None:
            theta = 1
        else:
            theta = argcheck.getunit(theta, units)

        if isinstance(theta, (int, np.int64, float, np.float64)):
            return SE3(tr.trexp(self.S *  theta))
        else:
            return SE3([tr.trexp(self.S *  t) for t in theta])

        
    def __str__(self):
        """
    %Twist.char Convert to string

    s = TW.char() is a string showing Twist parameters in a compact single line format.
    If TW is a vector of Twist objects return a string with one line per Twist.

    See also Twist.display.
        """
        return '\n'.join(["({:.5g} {:.5g} {:.5g}; {:.5g} {:.5g} {:.5g})".format(*list(tw.S)) for tw in self])

    def __repr__(self):
        """
        %Twist.display Display parameters
        %
L.display() displays the twist parameters in compact single line format.  If L is a
vector of Twist objects displays one line per element.
        %
Notes::
- This method is invoked implicitly at the command line when the result
  of an expression is a Twist object and the command has no trailing
  semicolon.
        %
See also Twist.char.
        """
        
        if len(self) == 1:
            return "Twist([{:.5g}, {:.5g}, {:.5g}, {:.5g}, {:.5g}, {:.5g}])".format(*list(self))
        else:
            return "Twist([\n" + \
                ',\n'.join(["  [{:.5g}, {:.5g}, {:.5g}, {:.5g}, {:.5g}, {:.5g}]".format(*list(tw)) for tw in self]) +\
                "\n])"

            
if __name__ == '__main__':   # pragma: no cover

    import pathlib
    import os.path
    
    x = Twist.P([1, 2, 3])

    a = Twist.isvalid(x.se3)
    print(a)
    a = Twist.isvalid(x.S)
    print(a)

    exec(open(os.path.join(pathlib.Path(__file__).parent.absolute(), "test_pose3d.py")).read())
