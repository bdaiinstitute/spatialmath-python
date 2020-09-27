#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Classes to abstract 3D pose and orientation using matrices in SE(3) and SO(3)

To use::

    from spatialmath.pose3d import *
    T = SE3.Rx(0.3)

    import spatialmath as sm
    T = sm.SE3.Rx(0.3)


 .. inheritance-diagram:: spatialmath.pose3d
    :top-classes: collections.UserList
    :parts: 1
"""

# pylint: disable=invalid-name

import numpy as np

from spatialmath.base import argcheck
import spatialmath.base as tr
from spatialmath.super_pose import SMPose


# ============================== SO3 =====================================#


class SO3(SMPose):  
    """
    SO(3) subclass

    This subclass represents rotations in 3D space.  Internally it is a 3x3 
    orthogonal matrix belonging to the group SO(3).

 .. inheritance-diagram:: spatialmath.pose3d.SO3
    :top-classes: collections.UserList
    :parts: 1
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
        - ``SO3([X1, X2, ... XN])`` where each Xi is an SO3 instance, is an SO3 instance containing N rotations.

        :seealso: `SMPose.pose_arghandler`
        """
        if not super().arghandler(arg, check=check):
            raise ValueError('bad argument to constructor')

    @staticmethod
    def _identity():
        return np.eye(3)
    # ------------------------------------------------------------------------ #
    @property
    def shape(self):
        """
        Shape of the object's interal matrix representation

        :return: (3,3)
        :rtype: tuple
        """
        return (3, 3)

    @property
    def R(self):
        """
        SO(3) or SE(3) as rotation matrix

        :return: rotational component
        :rtype: numpy.ndarray, shape=(3,3)

        ``x.R`` returns the rotation matrix, when `x` is `SO3` or `SE3`. If `len(x)` is:

        - 1, return an ndarray with shape=(3,3)
        - N>1, return ndarray with shape=(N,3,3)
        """
        if len(self) == 1:
            return self.A[:3, :3]
        else:
            return np.array([x[:3, :3] for x in self.A])

    @property
    def n(self):
        """
        Normal vector of SO(3) or SE(3)

        :return: normal vector
        :rtype: numpy.ndarray, shape=(3,)

        Is the first column of the rotation submatrix, sometimes called the normal
        vector.  Parallel to the x-axis of the frame defined by this pose.
        """
        return self.A[:3, 0]

    @property
    def o(self):
        """
        Orientation vector of SO(3) or SE(3)

        :return: orientation vector
        :rtype: numpy.ndarray, shape=(3,)

        Is the second column of the rotation submatrix, sometimes called the orientation
        vector.  Parallel to the y-axis of the frame defined by this pose.
        """
        return self.A[:3, 1]

    @property
    def a(self):
        """
        Approach vector of SO(3) or SE(3)

        :return: approach vector
        :rtype: numpy.ndarray, shape=(3,)

        Is the third column of the rotation submatrix, sometimes called the approach
        vector.  Parallel to the z-axis of the frame defined by this pose.
        """
        return self.A[:3, 2]

    # ------------------------------------------------------------------------ #

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
            return SO3(self.A.T, check=False)
        else:
            return SO3([x.T for x in self.A], check=False)

    def eul(self, unit='deg'):
        r"""
        SO(3) or SE(3) as Euler angles

        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :return: 3-vector of Euler angles
        :rtype: numpy.ndarray, shape=(3,)

        ``x.eul`` is the Euler angle representation of the rotation.  Euler angles are
        a 3-vector :math:`(\phi, \theta, \psi)` which correspond to consecutive
        rotations about the Z, Y, Z axes respectively.

        If `len(x)` is:

        - 1, return an ndarray with shape=(3,)
        - N>1, return ndarray with shape=(N,3)

        - ndarray with shape=(3,), if len(R) == 1
        - ndarray with shape=(N,3), if len(R) = N > 1

        :seealso: :func:`~spatialmath.pose3d.SE3.Eul`, ::func:`spatialmath.base.transforms3d.tr2eul`
        """
        if len(self) == 1:
            return tr.tr2eul(self.A, unit=unit)
        else:
            return np.array([tr.tr2eul(x, unit=unit) for x in self.A]).T

    def rpy(self, unit='deg', order='zyx'):
        """
        SO(3) or SE(3) as roll-pitch-yaw angles

        :param order: angle sequence order, default to 'zyx'
        :type order: str
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :return: 3-vector of roll-pitch-yaw angles
        :rtype: numpy.ndarray, shape=(3,)

        ``x.rpy`` is the roll-pitch-yaw angle representation of the rotation.  The angles are
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

        If `len(x)` is:

        - 1, return an ndarray with shape=(3,)
        - N>1, return ndarray with shape=(N,3)

        :seealso: :func:`~spatialmath.pose3d.SE3.RPY`, ::func:`spatialmath.base.transforms3d.tr2rpy`
        """
        if len(self) == 1:
            return tr.tr2rpy(self.A, unit=unit, order=order)
        else:
            return np.array([tr.tr2rpy(x, unit=unit, order=order) for x in self.A]).T

    def angvec(self, unit='rad'):
        r"""
        SO(3) or SE(3) as angle and rotation vector

        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :param check: check that rotation matrix is valid
        :type check: bool
        :return: :math:`(\theta, {\bf v})`
        :rtype: float, numpy.ndarray, shape=(3,)

        ``q.angvec()`` is a tuple :math:`(\theta, v)` containing the rotation 
        angle and a rotation axis which is equivalent to the rotation of
        the unit quaternion ``q``.

        By default the angle is in radians but can be changed setting `unit='deg'`.

        Notes:

        - If the input is SE(3) the translation component is ignored.

        Example::

        >>> UnitQuaternion.Rz(0.3).angvec()
            (0.3, array([0., 0., 1.]))

        :seealso: :func:`~spatialmath.quaternion.AngVec`, :func:`~angvec2r`
        """
        return tr.tr2angvec(self.R, unit=unit)

    def Ad(self):
        """
        Adjoint of SO(3)

        :return: adjoint matrix
        :rtype: numpy.ndarray, shape=(3,3)

        - ``SEO.Ad`` is the 6x6 adjoint matrix

        :seealso: Twist.ad.

        """
        return np.array([
            [self.R, tr.skew(self.t) @ self.R],
            [np.zeros((3, 3)), self.R]
            ])
    # ------------------------------------------------------------------------ #

    @staticmethod
    def isvalid(x, check=True):
        """
        Test if matrix is valid SO(3)

        :param x: matrix to test
        :type x: numpy.ndarray
        :return: true if the matrix is a valid element of SO(3), ie. it is a 3x3
            orthonormal matrix with determinant of +1.
        :rtype: bool

        :seealso: :func:`~spatialmath.base.transform3d.isrot`
        """
        return tr.isrot(x, check=True)

    # ---------------- variant constructors ---------------------------------- #

    @classmethod
    def Rx(cls, theta, unit='rad'):
        """
        Construct a new SO(3) from X-axis rotation

        :param θ: rotation angle about the X-axis
        :type θ: float or array_like
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :return: SO(3) rotation
        :rtype: SO3 instance

        - ``SE3.Rx(θ)`` is an SO(3) rotation of ``θ`` radians about the x-axis
        - ``SE3.Rx(θ, "deg")`` as above but ``θ`` is in degrees

        If ``theta`` is an array then the result is a sequence of rotations defined by consecutive
        elements.

        Example::

            >>> x = SO3.Rx(np.linspace(0, math.pi, 20))
            >>> len(x)
            20
            >>> x[7]
            SO3(array([[ 1.        ,  0.        ,  0.        ],
                       [ 0.        ,  0.40169542, -0.91577333],
                       [ 0.        ,  0.91577333,  0.40169542]]))
        """
        return cls([tr.rotx(x, unit=unit) for x in argcheck.getvector(theta)], check=False)

    @classmethod
    def Ry(cls, theta, unit='rad'):
        """
        Construct a new SO(3) from Y-axis rotation

        :param θ: rotation angle about Y-axis
        :type θ: float or array_like
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :return: SO(3) rotation
        :rtype: SO3 instance

        - ``SO3.Ry(θ)`` is an SO(3) rotation of ``θ`` radians about the y-axis
        - ``SO3.Ry(θ, "deg")`` as above but ``θ`` is in degrees

        If ``θ`` is an array then the result is a sequence of rotations defined by consecutive
        elements.

        Example::

            >>> x = SO3.Ry(np.linspace(0, math.pi, 20))
            >>> len(x)
            20
            >>> x[7]
            >>> x[7]
            SO3(array([[ 0.40169542,  0.        ,  0.91577333],
                       [ 0.        ,  1.        ,  0.        ],
                       [-0.91577333,  0.        ,  0.40169542]]))
        """
        return cls([tr.roty(x, unit=unit) for x in argcheck.getvector(theta)], check=False)

    @classmethod
    def Rz(cls, theta, unit='rad'):
        """
        Construct a new SO(3) from Z-axis rotation

        :param θ: rotation angle about Z-axis
        :type θ: float or array_like
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :return: SO(3) rotation
        :rtype: SO3 instance

        - ``SO3.Rz(θ)`` is an SO(3) rotation of ``θ`` radians about the z-axis
        - ``SO3.Rz(θ, "deg")`` as above but ``θ`` is in degrees

        If ``θ`` is an array then the result is a sequence of rotations defined by consecutive
        elements.

        Example::

            >>> x = SE3.Rz(np.linspace(0, math.pi, 20))
            >>> len(x)
            20
            SO3(array([[ 0.40169542, -0.91577333,  0.        ],
                       [ 0.91577333,  0.40169542,  0.        ],
                       [ 0.        ,  0.        ,  1.        ]]))
        """
        return cls([tr.rotz(x, unit=unit) for x in argcheck.getvector(theta)], check=False)

    @classmethod
    def Rand(cls, N=1):
        """
        Construct a new SO(3) from random rotation

        :param N: number of random rotations
        :type N: int
        :return: SO(3) rotation matrix
        :rtype: SO3 instance

        - ``SO3.Rand()`` is a random SO(3) rotation.
        - ``SO3.Rand(N)`` is a sequence of N random rotations.

        Example::

            >>> x = SO3.Rand()
            >>> x
            SO3(array([[ 0.1805082 , -0.97959019,  0.08842995],
                       [-0.98357187, -0.17961408,  0.01803234],
                       [-0.00178104, -0.0902322 , -0.99591916]]))

        :seealso: :func:`spatialmath.quaternion.UnitQuaternion.Rand`
        """
        return cls([tr.q2r(tr.rand()) for _ in range(0, N)], check=False)

    @classmethod
    def Eul(cls, angles, *, unit='rad'):
        r"""
        Construct a new SO(3) from Euler angles

        :param angles: Euler angles
        :type angles: array_like or numpy.ndarray with shape=(N,3)
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :return: SO(3) rotation
        :rtype: SO3 instance

        ``SO3.Eul(angles)`` is an SO(3) rotation defined by a 3-vector of Euler angles :math:`(\phi, \theta, \psi)` which
        correspond to consecutive rotations about the Z, Y, Z axes respectively.

        If ``angles`` is an Nx3 matrix then the result is a sequence of rotations each defined by Euler angles
        corresponding to the rows of ``angles``.

        :seealso: :func:`~spatialmath.pose3d.SE3.eul`, :func:`~spatialmath.pose3d.SE3.Eul`, :func:`spatialmath.base.transforms3d.eul2r`
        """
        if argcheck.isvector(angles, 3):
            return cls(tr.eul2r(angles, unit=unit), check=False)
        else:
            return cls([tr.eul2r(a, unit=unit) for a in angles], check=False)

    @classmethod
    def RPY(cls, angles, *, order='zyx', unit='rad'):
        r"""
        Construct a new SO(3) from roll-pitch-yaw angles

        :param angles: roll-pitch-yaw angles
        :type angles: array_like or numpy.ndarray with shape=(N,3)
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :param order: rotation order: 'zyx' [default], 'xyz', or 'yxz'
        :type order: str
        :return: SO(3) rotation
        :rtype: SO3 instance

        ``SO3.RPY(angles)`` is an SO(3) rotation defined by a 3-vector of roll, pitch, yaw angles :math:`(r, p, y)`
          which correspond to successive rotations about the axes specified by ``order``:

            - 'zyx' [default], rotate by yaw about the z-axis, then by pitch about the new y-axis,
              then by roll about the new x-axis.  Convention for a mobile robot with x-axis forward
              and y-axis sideways.
            - 'xyz', rotate by yaw about the x-axis, then by pitch about the new y-axis,
              then by roll about the new z-axis. Convention for a robot gripper with z-axis forward
              and y-axis between the gripper fingers.
            - 'yxz', rotate by yaw about the y-axis, then by pitch about the new x-axis,
              then by roll about the new z-axis. Convention for a camera with z-axis parallel
              to the optic axis and x-axis parallel to the pixel rows.

        If ``angles`` is an Nx3 matrix then the result is a sequence of rotations each defined by RPY angles
        corresponding to the rows of angles.

        :seealso: :func:`~spatialmath.pose3d.SE3.rpy`, :func:`~spatialmath.pose3d.SE3.RPY`, :func:`spatialmath.base.transforms3d.rpy2r`
        """
        if argcheck.isvector(angles, 3):
            return cls(tr.rpy2r(angles, order=order, unit=unit), check=False)
        else:
            return cls([tr.rpy2r(a, order=order, unit=unit) for a in angles], check=False)

    @classmethod
    def OA(cls, o, a):
        """
        Construct a new SO(3) from two vectors

        :param o: 3-vector parallel to Y- axis
        :type o: array_like
        :param a: 3-vector parallel to the Z-axis
        :type o: array_like
        :return: SO(3) rotation
        :rtype: SO3 instance

        ``SO3.OA(O, A)`` is an SO(3) rotation defined in terms of
        vectors parallel to the Y- and Z-axes of its reference frame.  In robotics these axes are
        respectively called the *orientation* and *approach* vectors defined such that
        R = [N, O, A] and N = O x A.

        Notes:

        - Only the ``A`` vector is guaranteed to have the same direction in the resulting
          rotation matrix
        - ``O`` and ``A`` do not have to be unit-length, they are normalized
        - ``O`` and ``A` do not have to be orthogonal, so long as they are not parallel

        :seealso: :func:`spatialmath.base.transforms3d.oa2r`
        """
        return cls(tr.oa2r(o, a), check=False)

    @classmethod
    def AngVec(cls, theta, v, *, unit='rad'):
        r"""
        Construct a new SO(3) rotation matrix from rotation angle and axis

        :param theta: rotation
        :type theta: float
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :param v: rotation axis, 3-vector
        :type v: array_like
        :return: SO(3) rotation
        :rtype: SO3 instance

        ``SO3.AngVec(theta, V)`` is an SO(3) rotation defined by
        a rotation of ``THETA`` about the vector ``V``.

        If :math:`\theta \eq 0` the result in an identity matrix, otherwise
        ``V`` must have a finite length, ie. :math:`|V| > 0`.

        :seealso: :func:`~spatialmath.pose3d.SE3.angvec`, :func:`spatialmath.base.transforms3d.angvec2r`
        """
        return cls(tr.angvec2r(theta, v, unit=unit), check=False)

    @classmethod
    def Exp(cls, S, check=True, so3=True):
        r"""
        Create an SO(3) rotation matrix from so(3)

        :param S: Lie algebra so(3)
        :type S: numpy ndarray
        :param check: check that passed matrix is valid so(3), default True
        :type check: bool
        :return: SO(3) rotation
        :rtype: SO3 instance

        - ``SO3.Exp(S)`` is an SO(3) rotation defined by its Lie algebra
          which is a 3x3 so(3) matrix (skew symmetric)
        - ``SO3.Exp(t)`` is an SO(3) rotation defined by a 3-element twist
          vector (the unique elements of the so(3) skew-symmetric matrix)
        - ``SO3.Exp(T)`` is a sequence of SO(3) rotations defined by an Nx3 matrix
          of twist vectors, one per row.

        Note:
        - if :math:`\theta \eq 0` the result in an identity matrix
        - an input 3x3 matrix is ambiguous, it could be the first or third case above.  In this
          case the parameter `so3` is the decider.

        :seealso: :func:`spatialmath.base.transforms3d.trexp`, :func:`spatialmath.base.transformsNd.skew`
        """
        if argcheck.ismatrix(S, (-1, 3)) and not so3:
            return cls([tr.trexp(s, check=check) for s in S], check=False)
        else:
            return cls(tr.trexp(S, check=check), check=False)

# ============================== SE3 =====================================#


class SE3(SO3):
    """
    SE(3) subclass

    This subclass represents rigid-body motion in 3D space.  Internally it is a 
    4x4 homogeneous transformation matrix belonging to the group SE(3).

 .. inheritance-diagram:: spatialmath.pose3d.SE3
    :top-classes: collections.UserList
    :parts: 1
    """

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
        - ``SE3([X1, X2, ... XN])`` where each Xi is an SE3 instance, is an SE3 instance containing N rotations.
        """
        if y is None and z is None:
            # just one argument passed

            if super().arghandler(x, check=check):
                return
            elif argcheck.isvector(x, 3):
                # SE3( [x, y, z] )
                self.data = [tr.transl(x)]
            elif isinstance(x, np.ndarray) and x.shape[1] == 3:
                # SE3( Nx3 )
                self.data = [tr.transl(T) for T in x]

            else:
                raise ValueError('bad argument to constructor')

        elif y is not None and z is not None:
            # SE3(x, y, z)
            self.data = [tr.transl(x, y, z)]

    @staticmethod
    def _identity():
        return np.eye(4)
        
    # ------------------------------------------------------------------------ #
    @property
    def shape(self):
        """
        Shape of the object's interal matrix representation

        :return: (4,4)
        :rtype: tuple
        """
        return (4, 4)

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

    # ------------------------------------------------------------------------ #

    def inv(self):
        r"""
        Inverse of SE(3)

        :return: inverse
        :rtype: SE3

        Returns the inverse taking into account its structure

        :math:`T = \left[ \begin{array}{cc} R & t \\ 0 & 1 \end{array} \right], T^{-1} = \left[ \begin{array}{cc} R^T & -R^T t \\ 0 & 1 \end{array} \right]`

        :seealso: :func:`~spatialmath.base.transform3d.trinv`
        """
        if len(self) == 1:
            return SE3(tr.trinv(self.A), check=False)
        else:
            return SE3([tr.trinv(x) for x in self.A], check=False)

    def delta(self, X2):
        r"""
        Infinitesimal difference of SE(3)

        :param X2:
        :type X2: SE3
        :return: differential motion vector
        :rtype: numpy.ndarray, shape=(6,)

        - ``X1.delta(X2)`` is the differential motion (6x1) corresponding to
          infinitesimal motion (in the X1 frame) from pose X1 to X2.

        The vector :math:`d = [\delta_x, \delta_y, \delta_z, \theta_x, \theta_y, \theta_z`
        represents infinitesimal translation and rotation.

        Notes:

        - the displacement is only an approximation to the motion T, and assumes
          that X1 ~ X2.
        - Can be considered as an approximation to the effect of spatial velocity over a
          a time interval, average spatial velocity multiplied by time.

        Reference: Robotics, Vision & Control: Second Edition, P. Corke, Springer 2016; p67.

        :seealso: :func:`~spatialmath.base.transform3d.tr2delta`
        """
        return tr.tr2delta(self.A, X2.A)

    def Twist3(self):

        from spatialmath.twist import Twist3

        return Twist3(self.log(twist=True))
    # ------------------------------------------------------------------------ #

    @staticmethod
    def isvalid(x, check=True):
        """
        Test if matrix is valid SE(3)

        :param x: matrix to test
        :type x: numpy.ndarray
        :return: true of the matrix is 4x4 and a valid element of SE(3), ie. it
        is an homogeneous transformation matrix.
        :rtype: bool

        :seealso: :func:`~spatialmath.base.transform3d.ishom`
        """
        return tr.ishom(x, check=check)

    # ---------------- variant constructors ---------------------------------- #

    @classmethod
    def Rx(cls, theta, unit='rad'):
        """
        Create SE(3) pure rotation about the X-axis

        :param θ: rotation angle about X-axis
        :type θ: float
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :return: 4x4 homogeneous transformation matrix
        :rtype: SE3 instance

        - ``SE3.Rx(θ)`` is an SO(3) rotation of θ radians about the x-axis
        - ``SE3.Rx(θ, "deg")`` as above but θ is in degrees

        If ``θ`` is an array then the result is a sequence of rotations defined
        by consecutive elements.
        """
        return cls([tr.trotx(x, unit) for x in argcheck.getvector(theta)], check=False)

    @classmethod
    def Ry(cls, theta, unit='rad'):
        """
        Create SE(3) pure rotation about the Y-axis

        :param θ: rotation angle about X-axis
        :type θ: float
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :return: 4x4 homogeneous transformation matrix
        :rtype: SE3 instance

        - ``SE3.Ry(θ)`` is an SO(3) rotation of θ radians about the y-axis
        - ``SE3.Ry(θ, "deg")`` as above but θ is in degrees

        If ``θ`` is an array then the result is a sequence of rotations defined
        by consecutive
        elements.
        """
        return cls([tr.troty(x, unit) for x in argcheck.getvector(theta)], check=False)

    @classmethod
    def Rz(cls, theta, unit='rad'):
        """
        Create SE(3) pure rotation about the Z-axis

        :param θ: rotation angle about Z-axis
        :type θ: float
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :return: 4x4 homogeneous transformation matrix
        :rtype: SE3 instance

        - ``SE3.Rz(θ)`` is an SO(3) rotation of θ radians about the z-axis
        - ``SE3.Rz(θ, "deg")`` as above but θ is in degrees

        If ``θ`` is an array then the result is a sequence of rotations defined
        by consecutive elements.
        """
        return cls([tr.trotz(x, unit) for x in argcheck.getvector(theta)], check=False)

    @classmethod
    def Rand(cls, *, xrange=(-1, 1), yrange=(-1, 1), zrange=(-1, 1), N=1):  # pylint: disable=arguments-differ
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
        return cls([tr.transl(x, y, z) @ tr.r2t(r.A) for (x, y, z, r) in zip(X, Y, Z, R)], check=False)

    @classmethod
    def Eul(cls, angles, *, unit='rad'):
        r"""
        Create an SE(3) pure rotation from Euler angles

        :param angles: 3-vector of Euler angles
        :type angles: array_like or numpy.ndarray with shape=(N,3)
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :return: 4x4 homogeneous transformation matrix
        :rtype: SE3 instance

        ``SE3.Eul(ANGLES)`` is an SO(3) rotation defined by a 3-vector of Euler
        angles :math:`(\phi, \theta, \psi)` which correspond to consecutive
        rotations about the Z, Y, Z axes respectively.

        If ``angles`` is an Nx3 matrix then the result is a sequence of
        rotations each defined by Euler angles corresponding to the rows of
        angles.

        :seealso: :func:`~spatialmath.pose3d.SE3.eul`, :func:`~spatialmath.pose3d.SE3.Eul`, :func:`spatialmath.base.transforms3d.eul2r`
        """
        if argcheck.isvector(angles, 3):
            return cls(tr.eul2tr(angles, unit=unit), check=False)
        else:
            return cls([tr.eul2tr(a, unit=unit) for a in angles], check=False)

    @classmethod
    def RPY(cls, angles, *, order='zyx', unit='rad'):
        """
        Create an SO(3) pure rotation from roll-pitch-yaw angles

        :param angles: 3-vector of roll-pitch-yaw angles
        :type angles: array_like or numpy.ndarray with shape=(N,3)
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :param order: rotation order: 'zyx' [default], 'xyz', or 'yxz'
        :type order: str
        :return: 4x4 homogeneous transformation matrix
        :rtype: SE3 instance

        ``SE3.RPY(ANGLES)`` is an SE(3) rotation defined by a 3-vector of roll, pitch, yaw angles :math:`(r, p, y)`
          which correspond to successive rotations about the axes specified by ``order``:

            - 'zyx' [default], rotate by yaw about the z-axis, then by pitch about the new y-axis,
              then by roll about the new x-axis.  Convention for a mobile robot with x-axis forward
              and y-axis sideways.
            - 'xyz', rotate by yaw about the x-axis, then by pitch about the new y-axis,
              then by roll about the new z-axis. Convention for a robot gripper with z-axis forward
              and y-axis between the gripper fingers.
            - 'yxz', rotate by yaw about the y-axis, then by pitch about the new x-axis,
              then by roll about the new z-axis. Convention for a camera with z-axis parallel
              to the optic axis and x-axis parallel to the pixel rows.

        If ``angles`` is an Nx3 matrix then the result is a sequence of rotations each defined by RPY angles
        corresponding to the rows of angles.

        :seealso: :func:`~spatialmath.pose3d.SE3.rpy`, :func:`~spatialmath.pose3d.SE3.RPY`, :func:`spatialmath.base.transforms3d.rpy2r`
        """
        if argcheck.isvector(angles, 3):
            return cls(tr.rpy2tr(angles, order=order, unit=unit), check=False)
        else:
            return cls([tr.rpy2tr(a, order=order, unit=unit) for a in angles], check=False)

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
        return cls(tr.oa2tr(o, a), check=False)

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
        return cls(tr.angvec2tr(theta, v, unit=unit), check=False)

    @classmethod
    def Exp(cls, S, check=True):
        """
        Create an SE(3) rotation matrix from se(3)

        :param S: Lie algebra se(3)
        :type S: numpy ndarray
        :return: 3x3 rotation matrix
        :rtype: SO3 instance

        - ``SE3.Exp(S)`` is an SE(3) rotation defined by its Lie algebra
          which is a 4x4 se(3) matrix (skew symmetric)
        - ``SE3.Exp(t)`` is an SE(3) rotation defined by a 6-element twist
          vector (the unique elements of the se(3) skew-symmetric matrix)

        :seealso: :func:`spatialmath.base.transforms3d.trexp`, :func:`spatialmath.base.transformsNd.skew`
        """
        if isinstance(S, list):
            return cls([tr.exp(s) for s in S], check=False)
        else:
            return cls(tr.trexp(S), check=False)

    @classmethod
    def Tx(cls, x):
        """
        Create SE(3) translation along the X-axis

        :param x: translation distance along the X-axis
        :type x: float
        :return: 4x4 homogeneous transformation matrix
        :rtype: SE3 instance

        `SE3.Tz(D)`` is an SE(3) translation of D along the x-axis
        """
        return cls(tr.transl(x, 0, 0), check=False)

    @classmethod
    def Ty(cls, y):
        """
        Create SE(3) translation along the Y-axis

        :param y: translation distance along the Y-axis
        :type y: float
        :return: 4x4 homogeneous transformation matrix
        :rtype: SE3 instance

        `SE3.Tz(D)`` is an SE(3) translation of D along the y-axis
        """
        return cls(tr.transl(0, y, 0), check=False)

    @classmethod
    def Tz(cls, z):
        """
        Create SE(3) translation along the Z-axis

        :param z: translation distance along the Z-axis
        :type z: float
        :return: 4x4 homogeneous transformation matrix
        :rtype: SE3 instance

        `SE3.Tz(D)`` is an SE(3) translation of D along the z-axis
        """
        return cls(tr.transl(0, 0, z), check=False)

    @classmethod
    def Delta(cls, d):
        r"""
        Create SE(3) from differential motion

        :param d: differential motion
        :type d: 6-element array_like
        :return: 4x4 homogeneous transformation matrix
        :rtype: SE3 instance


        - ``T = delta2tr(d)`` is an SE(3) representing differential 
          motion :math:`d = [\delta_x, \delta_y, \delta_z, \theta_x, \theta_y, \theta_z`.

        Reference: Robotics, Vision & Control: Second Edition, P. Corke, Springer 2016; p67.

        :seealso: :func:`~delta`, :func:`~spatialmath.base.transform3d.delta2tr`

        """
        return tr.tr2delta(d)


if __name__ == '__main__':   # pragma: no cover

    import pathlib

    exec(open(pathlib.Path(__file__).parent.absolute() / "test_pose3d.py").read())  # pylint: disable=exec-used
