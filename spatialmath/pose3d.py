#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 15:48:52 2020

@author: Peter Corke
"""

from collections import UserList
import numpy as np
import math

from spatialmath.base import argcheck 
import spatialmath.base as tr
from spatialmath import super_pose as sp


    
class SO3(sp.SMPose):
    
    # SO2()  identity matrix
    # SO2(angle, unit)
    # SO2( obj )   # deep copy
    # SO2( np )  # make numpy object
    # SO2( nplist )  # make from list of numpy objects
    
    # constructor needs to take ndarray -> SO2, or list of ndarray -> SO2
    def __init__(self, arg = None, *, check=True):
        super().__init__()  # activate the UserList semantics
        
        if arg is None:
            # empty constructor
            if type(self)  == SO3:
                self.data = [np.eye(3)]
        else:
            super().pose_arghandler(arg, check=check)
    

    @property
    def inv(self):
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
        return self.A[:3,0]
       
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
        return self.A[:3,1]
        
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
        return self.A[:3,2]

    @classmethod
    def isvalid(cls, x):
        """
        Test if matrix is valid SO(3)

        :param x: matrix to test
        :type x: numpy.ndarray
        :return: true of the matrix is 3x3 and a valid element of SO(3), ie. it is an
        orthonormal matrix with determinant of +1.
        
        :seealso: :func:`~spatialmath.base.transform3d.isrot`
        """
        return tr.isrot(x, check=True)

        
    @classmethod
    def Rx(cls, theta, unit='rad'):
        """
        Create SO(3) rotation about X-axis
    
        :param theta: rotation angle about the X-axis
        :type theta: float
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :return: 3x3 rotation matrix
        :rtype: SO3 instance
    
        - ``SE3.Rx(THETA)`` is an SO(3) rotation of THETA radians about the x-axis
        - ``SE3.Rx(THETA, "deg")`` as above but THETA is in degrees
        """
        return cls([tr.rotx(x, unit=unit) for x in argcheck.getvector(theta)], check=False)

    @classmethod
    def Ry(cls, theta, unit='rad'):
        """
        Create SO(3) rotation about the Y-axis
    
        :param theta: rotation angle about Y-axis
        :type theta: float
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :return: 3x3 rotation matrix
        :rtype: SO3 instance
    
        - ``SO3.Ry(THETA)`` is an SO(3) rotation of THETA radians about the y-axis
        - ``SO3.Ry(THETA, "deg")`` as above but THETA is in degrees
        """
        return cls([tr.roty(x, unit=unit) for x in argcheck.getvector(theta)], check=False)

    @classmethod
    def Rz(cls, theta, unit='rad'):
        """
        Create SO(3) rotation about the Z-axis
    
        :param theta: rotation angle about Z-axis
        :type theta: float
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :return: 3x3 rotation matrix
        :rtype: SO3 instance
    
        - ``SO3.Rz(THETA)`` is an SO(3) rotation of THETA radians about the z-axis
        - ``SO3.Rz(THETA, "deg")`` as above but THETA is in degrees
        """
        return cls([tr.rotz(x, unit=unit) for x in argcheck.getvector(theta)], check=False)

    @classmethod
    def Rand(cls, N=1):
        """
        Create random SO(3) rotation
    
        :param N: number of random rotations
        :type N: int
        :return: 3x3 rotation matrix
        :rtype: SO3 instance
    
        - ``SO3.Rand()`` is a random SO(3) rotation.
        - ``SO3.Rand(N)`` is an SO3 object containing a sequence of N random
          rotations.
        
        :seealso: `~spatialmath.quaternion.UnitQuaternion.Rand`
        """
        return cls( [tr.q2r(tr.rand()) for i in range(0,N)], check=False)
        

    # 
    

    @classmethod
    def Eul(cls, angles, *, unit='rad'):
        """
        Create an SO(3) rotation matrix from Euler angles
    
        :param angles: 3-vector of Euler angles
        :type angles: array_like
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :return: 3x3 rotation matrix
        :rtype: SO3 instance
    
        - ``R = eul2r(PHI, THETA, PSI)`` is an SO(3) orthonornal rotation
          matrix equivalent to the specified Euler angles.  These correspond
          to rotations about the Z, Y, Z axes respectively.
        - ``R = eul2r(EUL)`` as above but the Euler angles are taken from
          ``EUL`` which is a 3-vector (array_like) with values
          (PHI THETA PSI).
          
        :seealso: :func:`~rpy2r`, :func:`~eul2tr`, :func:`~tr2eul`
        """
        return cls(tr.eul2r(angles, unit=unit), check=False)

    @classmethod
    def RPY(cls, angles, *, order='zyx', unit='rad'):
        """
        Create an SO(3) rotation matrix from roll-pitch-yaw angles
    
        :param angles: 3-vector of roll-pitch-yaw angles
        :type angles: array_like
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :param unit: rotation order: 'zyx' [default], 'xyz', or 'yxz'
        :type unit: str
        :return: 3x3 rotation matrix
        :rtype: SO3 instance
    
        - ``rpy2r(ROLL, PITCH, YAW)`` is an SO(3) orthonormal rotation matrix
          (3x3) equivalent to the specified roll, pitch, yaw angles angles.
          These correspond to successive rotations about the axes specified by ``order``:
              
            - 'zyx' [default], rotate by yaw about the z-axis, then by pitch about the new y-axis,
              then by roll about the new x-axis.  Convention for a mobile robot with x-axis forward
              and y-axis sideways.
            - 'xyz', rotate by yaw about the x-axis, then by pitch about the new y-axis,
              then by roll about the new z-axis. Covention for a robot gripper with z-axis forward
              and y-axis between the gripper fingers.
            - 'yxz', rotate by yaw about the y-axis, then by pitch about the new x-axis,
              then by roll about the new z-axis. Convention for a camera with z-axis parallel
              to the optic axis and x-axis parallel to the pixel rows.
              
        - ``rpy2r(RPY)`` as above but the roll, pitch, yaw angles are taken
          from ``RPY`` which is a 3-vector (array_like) with values
          (ROLL, PITCH, YAW).
          
        :seealso: :func:`~eul2r`, :func:`~rpy2tr`, :func:`~tr2rpy`
        """
        return cls(tr.rpy2r(angles, order=order, unit=unit), check=False)

    @classmethod
    def OA(cls, o, a):
        """
        Create SO(3) rotation matrix from two vectors
    
        :param o: 3-vector parallel to Y- axis
        :type o: array_like
        :param a: 3-vector parallel to the Z-axis
        :type o: array_like
        :return: 3x3 rotation matrix
        :rtype: numpy.ndarray, shape=(3,3)
    
        ``T = oa2tr(O, A)`` is an SO(3) orthonormal rotation matrix for a frame defined in terms of
        vectors parallel to its Y- and Z-axes with respect to a reference frame.  In robotics these axes are 
        respectively called the orientation and approach vectors defined such that
        R = [N O A] and N = O x A.
        
        Steps:
            
            1. N' = O x A
            2. O' = A x N
            3. normalize N', O', A
            4. stack horizontally into rotation matrix
    
        Notes:
            
        - The A vector is the only guaranteed to have the same direction in the resulting 
          rotation matrix
        - O and A do not have to be unit-length, they are normalized
        - O and A do not have to be orthogonal, so long as they are not parallel
        - The vectors O and A are parallel to the Y- and Z-axes of the equivalent coordinate frame.
    
        :seealso: :func:`~oa2tr`
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
        :rtype: numpdy.ndarray, shape=(3,3)
        
        ``angvec2r(THETA, V)`` is an SO(3) orthonormal rotation matrix
        equivalent to a rotation of ``THETA`` about the vector ``V``.
        
        Notes:
            
        - If ``THETA == 0`` then return identity matrix.
        - If ``THETA ~= 0`` then ``V`` must have a finite length.
    
        :seealso: :func:`~angvec2tr`, :func:`~tr2angvec`
        """
        return cls(tr.angvec2r(theta, v, unit=unit), check=False)

class SE3(SO3):

    def __init__(self, arg = None, *, unit='rad', check=True):
        super().__init__()  # activate the UserList semantics
        
        if arg is None:
            # empty constructor
            self.data = [np.eye(4)]
        else:
            super().pose_arghandler(arg, check=check)

    @property
    def t(self):
        return self.A[:3,3]
    
    @property
    def R(self):
        return self.A[:3,:3]
    
    @property
    def eul(self, **kwargs):
        return tr.tr2eul(self.A)
    
    @property
    def rpy(self, **kwargs):
        return tr.tr2eul(self.A, **kwargs)
    
    
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

        Computes the inverse taking into account its structure
        
        :math:`T = \left[ \begin{array}{cc} R & t \\ 0 & 1 \end{array} \right], T^{-1} = \left[ \begin{array}{cc} R^T & -R^T t \\ 0 & 1 \end{array} \right]`
        """
        if len(self) == 1:
            return SE3(tr.rt2tr(self.R.T, -self.R.T @ self.t))
        else:
            return SE3([SE3(tr.rt2tr(x.R.T, -x.R.T @ x.t)) for x in self])    
    
    @classmethod
    def isvalid(self, x):
        """
        Test if matrix is valid SE(3)

        :param x: matrix to test
        :type x: numpy.ndarray
        :return: true of the matrix is 4x4 and a valid element of SE(3), ie. it is an
        homogeneous transformation matrix.
        
        :seealso: :func:`~spatialmath.base.transform3d.ishom`
        """
        return tr.ishom(x, check=True)
    
    @classmethod
    def Rx(cls, theta, unit='rad'):
        """
        Create SE(3) rotation about the X-axis
    
        :param theta: rotation angle about X-axis
        :type theta: float
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :return: 4x4 homogeneous transformation matrix
        :rtype: SE3 instance
    
        - ``SE3.Rx(THETA)`` is an SO(3) rotation of THETA radians about the x-axis
        - ``SE3.Rx(THETA, "deg")`` as above but THETA is in degrees
        """
        return cls([tr.trotx(x, unit) for x in argcheck.getvector(theta)])

    @classmethod
    def Ry(cls, theta, unit='rad'):
        """
        Create SE(3) rotation about the Y-axis
    
        :param theta: rotation angle about X-axis
        :type theta: float
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :return: 4x4 homogeneous transformation matrix
        :rtype: SE3 instance
    
        - ``SE3.Ry(THETA)`` is an SO(3) rotation of THETA radians about the y-axis
        - ``SE3.Ry(THETA, "deg")`` as above but THETA is in degrees
        """
        return cls([tr.troty(x, unit) for x in argcheck.getvector(theta)])

    @classmethod
    def Rz(cls, theta, unit='rad'):
        """
        Create SE(3) rotation about the Z-axis
    
        :param theta: rotation angle about Z-axis
        :type theta: float
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :return: 4x4 homogeneous transformation matrix
        :rtype: SE3 instance
    
        - ``SE3.Rz(THETA)`` is an SO(3) rotation of THETA radians about the z-axis
        - ``SE3.Rz(THETA, "deg")`` as above but THETA is in degrees
        """
        return cls([tr.trotz(x, unit) for x in argcheck.getvector(theta)])
    
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
    
    @classmethod
    def trans(cls, x = None, y = None, z = None):
        """
        Create SE(3) general translation
    
        :param x: translation distance along the X-axis
        :type x: float
        :param y: translation distance along the Y-axis
        :type y: float
        :param z: translation distance along the Z-axis
        :type z: float
        :return: 4x4 homogeneous transformation matrix
        :rtype: SE3 instance
    
        - `SE3.Tz(X, Y, Z)`` is an SE(3) translation of X along the x-axis, Y along the
          y-axis and Z along the z-axis.
        - `SE3.Tz( [X, Y, Z] )`` as above but the translation is a 3-element array_like object.
        """
        return cls(tr.transl(x, y, z))
    

    

    

    

    @classmethod
    def Rand(cls, *, xrange=[-1, 1], yrange=[-1, 1], zrange=[-1, 1], N=1):
        X = np.random.uniform(low=xrange[0], high=xrange[1], size=N)  # random values in the range
        Y = np.random.uniform(low=yrange[0], high=yrange[1], size=N)  # random values in the range
        Z = np.random.uniform(low=yrange[0], high=zrange[1], size=N)  # random values in the range
        R = SO3.Rand(N=N)
        return cls([tr.transl(x, y, z) @ tr.r2t(r.A) for (x,y,z,r) in zip(X, Y, Z, R)])

    @classmethod
    def Eul(cls, angles, unit='rad'):
        return cls(tr.eul2tr(angles, unit=unit))

    @classmethod
    def RPY(cls, angles, order='zyx', unit='rad'):
        return cls(tr.rpy2tr(angles, order=order, unit=unit))

    @classmethod
    def OA(cls, o, a):
        return cls(tr.oa2tr(o, a))

    @classmethod
    def AngVec(cls, theta, v, *, unit='rad'):
        return cls(tr.angvec2tr(theta, v, unit=unit))

if __name__ == '__main__':   # pragma: no cover
    
    import pathlib
    import os.path
    
    runfile(os.path.join(pathlib.Path(__file__).parent.absolute(), "test_pose3d.py") )
