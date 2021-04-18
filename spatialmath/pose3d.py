# Part of Spatial Math Toolbox for Python
# Copyright (c) 2000 Peter Corke
# MIT Licence, see details in top-level file: LICENCE

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
    
.. image:: ../figs/pose-values.png
"""

# pylint: disable=invalid-name

import numpy as np

from spatialmath import base
from spatialmath.baseposematrix import BasePoseMatrix


# ============================== SO3 =====================================#


class SO3(BasePoseMatrix):  
    """
    SO(3) matrix class

    This subclass represents rotations in 3D space.  Internally it is a 3x3 
    orthogonal matrix belonging to the group SO(3).

 .. inheritance-diagram:: spatialmath.pose3d.SO3
    :top-classes: collections.UserList
    :parts: 1
    """

    def __init__(self, arg=None, *, check=True):
        """
        Construct new SO(3) object

        :rtype: SO3 instance

        There are multiple call signatures:

        - ``SO3()`` is an ``SO3`` instance with one value -- a 3x3 identity
          matrix which corresponds to a null rotation
        - ``SO3(R)`` is an ``SO3`` instance with with the value ``R`` which is a
          3x3 numpy array representing an SO(3) rotation matrix.  If ``check``
          is ``True`` check the matrix belongs to SO(3).
        - ``SO3([R1, R2, ... RN])`` is an ``SO3`` instance wwith ``N`` values
          given by the elements ``Ri`` each of which is a 3x3 NumPy array
          representing an SO(3) matrix. If ``check`` is ``True`` check the
          matrix belongs to SO(3).
        - ``SO3([X1, X2, ... XN])`` is an ``SO3`` instance with ``N`` values
          given by the elements ``Xi`` each of which is an SO3 instance.

        :SymPy: supported
        """
        super().__init__()
        
        if isinstance(arg, SE3):
            self.data = [base.t2r(x) for x in arg.data]

        elif not super().arghandler(arg, check=check):
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

        Each value within the ``SO3`` instance is a NumPy array of this shape.
        """
        return (3, 3)

    @property
    def R(self):
        """
        SO(3) or SE(3) as rotation matrix

        :return: rotational component
        :rtype: numpy.ndarray, shape=(3,3)

        ``x.R`` is the rotation matrix component of ``x`` as an array with
        shape (3,3). If ``len(x) > 1``, return an array with shape=(N,3,3).

        .. warning:: The i'th rotation matrix is ``x[i,:,:]`` or simply 
            ``x[i]``. This is different to the MATLAB version where the i'th
            rotation matrix is ``x(:,:,i)``.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import SO3
            >>> x = SO3.Rx(0.3)
            >>> x.R

        :SymPy: supported
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

        This is the first column of the rotation submatrix, sometimes called the
        *normal vector*.  It is parallel to the x-axis of the frame defined by
        this pose.
        """
        return self.A[:3, 0]

    @property
    def o(self):
        """
        Orientation vector of SO(3) or SE(3)

        :return: orientation vector
        :rtype: numpy.ndarray, shape=(3,)

        This is the second column of the rotation submatrix, sometimes called
        the *orientation vector*.  It is parallel to the y-axis of the frame
        defined by this pose.
        """
        return self.A[:3, 1]

    @property
    def a(self):
        """
        Approach vector of SO(3) or SE(3)

        :return: approach vector
        :rtype: numpy.ndarray, shape=(3,)

        This is the third column of the rotation submatrix, sometimes called the
        *approach vector*.  It is parallel to the z-axis of the frame defined by
        this pose.
        """
        return self.A[:3, 2]

    # ------------------------------------------------------------------------ #

    def inv(self):
        """
        Inverse of SO(3)

        :return: inverse
        :rtype: SO2 instance

        Efficiently compute the inverse of each of the SO(3) values taking into
        account the matrix structure.  For an SO(3) matrix the inverse is the
        transpose.
        """
        if len(self) == 1:
            return SO3(self.A.T, check=False)
        else:
            return SO3([x.T for x in self.A], check=False)

    def eul(self, unit='rad', flip=False):
        r"""
        SO(3) or SE(3) as Euler angles

        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :return: 3-vector of Euler angles
        :rtype: ndarray(3,), ndarray(n,3)

        ``x.eul`` is the Euler angle representation of the rotation.  Euler angles are
        a 3-vector :math:`(\phi, \theta, \psi)` which correspond to consecutive
        rotations about the Z, Y, Z axes respectively.

        If ``len(x)`` is:

        - 1, return an ndarray with shape=(3,)
        - N>1, return ndarray with shape=(3,N)

        :seealso: :func:`~spatialmath.pose3d.SE3.Eul`, :func:`~spatialmath.base.transforms3d.tr2eul`
        :SymPy: not supported
        """
        if len(self) == 1:
            return base.tr2eul(self.A, unit=unit, flip=flip)
        else:
            return np.array([base.tr2eul(x, unit=unit, flip=flip) for x in self.A])

    def rpy(self, unit='rad', order='zyx'):
        """
        SO(3) or SE(3) as roll-pitch-yaw angles

        :param order: angle sequence order, default to 'zyx'
        :type order: str
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :return: 3-vector of roll-pitch-yaw angles
        :rtype: ndarray(3,), ndarray(n,3)

        ``x.rpy`` is the roll-pitch-yaw angle representation of the rotation.  The angles are
        a 3-vector :math:`(r, p, y)` which correspond to successive rotations about the axes
        specified by ``order``:

            - ``'zyx'`` [default], rotate by yaw about the z-axis, then by pitch about the new y-axis,
              then by roll about the new x-axis.  Convention for a mobile robot with x-axis forward
              and y-axis sideways.
            - ``'xyz'``, rotate by yaw about the x-axis, then by pitch about the new y-axis,
              then by roll about the new z-axis. Convention for a robot gripper with z-axis forward
              and y-axis between the gripper fingers.
            - ``'yxz'``, rotate by yaw about the y-axis, then by pitch about the new x-axis,
              then by roll about the new z-axis. Convention for a camera with z-axis parallel
              to the optic axis and x-axis parallel to the pixel rows.

        If `len(x)` is:

        - 1, return an ndarray with shape=(3,)
        - N>1, return ndarray with shape=(3,N)

        :seealso: :func:`~spatialmath.pose3d.SE3.RPY`, :func:`~spatialmath.base.transforms3d.tr2rpy`
        :SymPy: not supported
        """
        if len(self) == 1:
            return base.tr2rpy(self.A, unit=unit, order=order)
        else:
            return np.array([base.tr2rpy(x, unit=unit, order=order) for x in self.A])

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

        .. notes::

            - If the input is SE(3) the translation component is ignored.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import UnitQuaternion
            >>> UnitQuaternion.Rz(0.3).angvec()

        :seealso: :func:`~spatialmath.quaternion.AngVec`, :func:`~angvec2r`
        """
        return base.tr2angvec(self.R, unit=unit)

    # ------------------------------------------------------------------------ #

    @staticmethod
    def isvalid(x, check=True):
        """
        Test if matrix is valid SO(3)

        :param x: matrix to test
        :type x: numpy.ndarray
        :return: ``True`` if the matrix is a valid element of SO(3), ie. it is a 3x3
            orthonormal matrix with determinant of +1.
        :rtype: bool

        :seealso: :func:`~spatialmath.base.transform3d.isrot`
        """
        return base.isrot(x, check=True)

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

        Example:

        .. runblock:: pycon

            >>> from spatialmath import SO3
            >>> x = SO3.Rx(np.linspace(0, math.pi, 20))
            >>> len(x)
            >>> x[7]

        """
        return cls([base.rotx(x, unit=unit) for x in base.getvector(theta)], check=False)

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

        Example:

        .. runblock:: pycon

            >>> from spatialmath import UnitQuaternion
            >>> x = SO3.Ry(np.linspace(0, math.pi, 20))
            >>> len(x)
            >>> x[7]

        """
        return cls([base.roty(x, unit=unit) for x in base.getvector(theta)], check=False)

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

        Example:

        .. runblock:: pycon

            >>> from spatialmath import SE3
            >>> x = SE3.Rz(np.linspace(0, math.pi, 20))
            >>> len(x)
            >>> x[7]

        """
        return cls([base.rotz(x, unit=unit) for x in base.getvector(theta)], check=False)

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

        Example:

        .. runblock:: pycon

            >>> from spatialmath import SO3
            >>> x = SO3.Rand()
            >>> x

        :seealso: :func:`spatialmath.quaternion.UnitQuaternion.Rand`
        """
        return cls([base.q2r(base.rand()) for _ in range(0, N)], check=False)

    @classmethod
    def Eul(cls, *angles, unit='rad'):
        r"""
        Construct a new SO(3) from Euler angles

        :param 𝚪: Euler angles
        :type 𝚪: array_like or numpy.ndarray with shape=(N,3)
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :return: SO(3) rotation
        :rtype: SO3 instance

        ``SO3.Eul(𝚪)`` is an SO(3) rotation defined by a 3-vector of Euler
          angles :math:`\Gamma = (\phi, \theta, \psi)` which correspond to
          consecutive rotations about the Z, Y, Z axes respectively. If ``𝚪``
          is an Nx3 matrix then the result is a sequence of rotations each
          defined by Euler angles corresponding to the rows of ``angles``.

        ``SO3.Eul(φ, θ, ψ)`` as above but the angles are provided as three
          scalars.

        Example:

        .. runblock:: pycon
        
            >>> from spatialmath import SO3
            >>> SO3.Eul(0.1, 0.2, 0.3)
            >>> SO3.Eul([0.1, 0.2, 0.3])
            >>> SO3.Eul(10, 20, 30, 'deg')

        :seealso: :func:`~spatialmath.pose3d.SE3.eul`, :func:`~spatialmath.pose3d.SE3.Eul`, :func:`~spatialmath.base.transforms3d.eul2r`
        """
        if len(angles) == 1:
            angles = angles[0]

        if base.isvector(angles, 3):
            return cls(base.eul2r(angles, unit=unit), check=False)
        else:
            return cls([base.eul2r(a, unit=unit) for a in angles], check=False)

    @classmethod
    def RPY(cls, *angles, unit='rad', order='zyx', ):
        r"""
        Construct a new SO(3) from roll-pitch-yaw angles

        :param angles: roll-pitch-yaw angles
        :type angles: array_like(3), array_like(n,3)
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :param order: rotation order: 'zyx' [default], 'xyz', or 'yxz'
        :type order: str
        :return: SO(3) rotation
        :rtype: SO3 instance

        - ``SO3.RPY(angles)`` is an SO(3) rotation defined by a 3-vector of
          roll, pitch, yaw angles :math:`(\alpha, \beta, \gamma)`. If ``angles``
          is an Nx3 matrix then the result is a sequence of rotations each
          defined by RPY angles corresponding to the rows of angles. The angles
          correspond to successive rotations about the axes specified by
          ``order``:

             - ``'zyx'`` [default], rotate by yaw about the z-axis, then by pitch about the new y-axis,
               then by roll about the new x-axis.  Convention for a mobile robot with x-axis forward
               and y-axis sideways.
            - ``'xyz'``, rotate by yaw about the x-axis, then by pitch about the new y-axis,
              then by roll about the new z-axis. Convention for a robot gripper with z-axis forward
              and y-axis between the gripper fingers.
            - ``'yxz'``, rotate by yaw about the y-axis, then by pitch about the new x-axis,
              then by roll about the new z-axis. Convention for a camera with z-axis parallel
              to the optic axis and x-axis parallel to the pixel rows.

        - ``SO3.RPY(⍺, β, 𝛾)`` as above but the angles are provided as three
          scalars.

        Example:

        .. runblock:: pycon
        
            >>> from spatialmath import SO3
            >>> SO3.RPY(0.1, 0.2, 0.3)
            >>> SO3.RPY([0.1, 0.2, 0.3])
            >>> SO3.RPY(0.1, 0.2, 0.3, order='xyz')
            >>> SO3.RPY(10, 20, 30, 'deg')


        :seealso: :func:`~spatialmath.pose3d.SE3.rpy`, :func:`~spatialmath.pose3d.SE3.RPY`, :func:`spatialmath.base.transforms3d.rpy2r`
        """
        if len(angles) == 1:
            angles = angles[0]

        # angles = base.getmatrix(angles, (None, 3))
        # return cls(base.rpy2r(angles, order=order, unit=unit), check=False)

        if base.isvector(angles, 3):
            return cls(base.rpy2r(angles, unit=unit, order=order), check=False)
        else:
            return cls([base.rpy2r(a, unit=unit, order=order) for a in angles], check=False)

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

        .. notes::

            - Only the ``A`` vector is guaranteed to have the same direction in the resulting
            rotation matrix
            - ``O`` and ``A`` do not have to be unit-length, they are normalized
            - ``O`` and ``A` do not have to be orthogonal, so long as they are not parallel

        :seealso: :func:`spatialmath.base.transforms3d.oa2r`
        """
        return cls(base.oa2r(o, a), check=False)

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

        .. note:: :math:`\theta \eq 0` the result in an identity matrix, otherwise
            ``V`` must have a finite length, ie. :math:`|V| > 0`.

        :seealso: :func:`~spatialmath.pose3d.SE3.angvec`, :func:`spatialmath.base.transforms3d.angvec2r`
        """
        return cls(base.angvec2r(theta, v, unit=unit), check=False)

    @classmethod
    def EulerVec(cls, w):
        r"""
        Construct a new SO(3) rotation matrix from an Euler rotation vector

        :param ω: rotation axis
        :type ω: 3-element array_like
        :return: SO(3) rotation
        :rtype: SO3 instance

        ``SO3.EulerVec(ω)`` is a unit quaternion that describes the 3D rotation
        defined by a rotation of :math:`\theta = \lVert \omega \rVert` about the
        unit 3-vector :math:`\omega / \lVert \omega \rVert`.

        Example:

        .. runblock:: pycon
        
            >>> from spatialmath import SO3
            >>> SO3.EulerVec([0.5,0,0])

        .. note:: :math:`\theta \eq 0` the result in an identity matrix, otherwise
            ``V`` must have a finite length, ie. :math:`|V| > 0`.

        :seealso: :func:`~spatialmath.pose3d.SE3.angvec`, :func:`~spatialmath.base.transforms3d.angvec2r`
        """
        assert base.isvector(w, 3), 'w must be a 3-vector'
        w = base.getvector(w)
        theta = base.norm(w)
        return cls(base.angvec2r(theta, w), check=False)

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
        if base.ismatrix(S, (-1, 3)) and not so3:
            return cls([base.trexp(s, check=check) for s in S], check=False)
        else:
            return cls(base.trexp(S, check=check), check=False)

    def angdist(self, other, metric=6):
        r"""
        Angular distance metric between rotations

        :param other: second rotation
        :type other: SO3 instance
        :param metric: metric, default is 6
        :type metric: int
        :raises TypeError: if other is not an SO3
        :return: angle in radians
        :rtype: float or ndarray

        ``R1.angdist(R2)`` is the geodesic norm, or geodesic distance between two
        rotations.

        Several metrics are supported, the first 5 are computed after conversion
        to unit quaternions.

        ======   ===============================================================
        Metric   Details
        ======   ===============================================================
        0        :math:`1 - | \q_1 \bullet \q_2 | \in [0, 1]`
        1        :math:`\cos^{-1} | \q_1 \bullet \q_2 | \in [0, \pi/2]`
        2        :math:`\cos^{-1} | \q_1 \bullet \q_2 | \in [0, \pi/2]`
        3        :math:`2 \tan^{-1} \| \q_1 - \q_2\| / \|\q_1 + \q_2\| \in [0, \pi/2]`
        4        :math:`\cos^{-1} \left( 2 (\q_1 \bullet \q_2)^2 - 1\right) \in [0, 1]`
        5        :math:`\|I - \mat{R}_1 \mat{R}_2^T\| \in [0, 2]`
        6        :math:`\|\log \mat{R}_1 \mat{R}_2^T\| \in [0, \pi]`
        ======   ===============================================================

        Example:

        .. runblock:: pycon

            >>> from spatialmath import UnitQuaternion
            >>> R1 = SO3.Rx(0.3)
            >>> R2 = SO3.Ry(0.3)
            >>> print(R1.angdist(R1))
            >>> print(R1.angdist(R2))

        .. note::
            - metrics 1, 2, 4 can throw ValueError "math domain error" due to
              numeric errors which push the argument of ``acos()`` marginally
              outside its domain [0, 1].
            - metrics 2 and 3 are equivalent, but 3 is more robust

        :seealso: :func:`UnitQuaternion.angdist`
        """

        if metric < 5:
            from spatialmath.quaternion import UnitQuaternion

            return UnitQuaternion(self).angdist(UnitQuaternion(other), metric=metric)

        elif metric == 5:
            op = lambda R1, R2: np.linalg.norm(np.eye(3) - R1 @ R2.T)
        elif metric == 6:
            op = lambda R1, R2: base.norm(base.trlog(R1 @ R2.T, twist=True))
        else:
            raise ValueError('unknown metric')
        
        ad = self._op2(other, op)
        if isinstance(ad, list):
            return np.array(ad)
        else:
            return ad

# ============================== SE3 =====================================#


class SE3(SO3):
    """
    SE(3) matrix class

    This subclass represents rigid-body motion in 3D space.  Internally it is a 
    4x4 homogeneous transformation matrix belonging to the group SE(3).

 .. inheritance-diagram:: spatialmath.pose3d.SE3
    :top-classes: collections.UserList
    :parts: 1
    """

    def __init__(self, x=None, y=None, z=None, *, check=True):
        """
        Construct new SE(3) object

        :rtype: SE3 instance

        There are multiple call signatures that return an ``SE3`` instance
        with one or more values.

        - ``SE3()`` null motion, value is the identity matrix.
        - ``SE3(x, y, z)`` is a pure translation of (x,y,z)
        - ``SE3(T)``  where ``T`` is a 4x4 Numpy  array representing an SE(3)
          matrix.  If ``check`` is ``True`` check the matrix belongs to SE(3).
        - ``SE3([T1, T2, ... TN])`` has ``N`` values
          given by the elements ``Ti`` each of which is a 4x4 NumPy array
          representing an SE(3) matrix. If ``check`` is ``True`` check the
          matrix belongs to SE(3).
        - ``SE3(X)`` where ``X`` is:
          -  ``SE3`` is a copy of ``X``
          -  ``SO3`` is the rotation of ``X`` with zero translation
          -  ``SE2`` is the z-axis rotation and x- and y-axis translation of
             ``X``
        - ``SE3([X1, X2, ... XN])`` has ``N`` values
          given by the elements ``Xi`` each of which is an SE3 instance.
        
        :SymPy: supported
        """
        if y is None and z is None:
            # just one argument passed

            if super().arghandler(x, check=check):
                return
            elif isinstance(x, SO3):
                self.data = [base.r2t(_x) for _x in x.data]
            elif type(x).__name__ == 'SE2':
                def convert(x):
                    # convert SE(2) to SE(3)
                    out = np.identity(4, dtype=x.dtype)
                    out[:2,:2] = x[:2,:2]
                    out[:2,3] = x[:2,2]
                    return out
                self.data = [convert(_x) for _x in x.data]
            elif base.isvector(x, 3):
                # SE3( [x, y, z] )
                self.data = [base.transl(x)]
            elif isinstance(x, np.ndarray) and x.shape[1] == 3:
                # SE3( Nx3 )
                self.data = [base.transl(T) for T in x]

            else:
                raise ValueError('bad argument to constructor')

        elif y is not None and z is not None:
            # SE3(x, y, z)
            self.data = [base.transl(x, y, z)]

    @staticmethod
    def _identity():
        return np.eye(4)
        
    # ------------------------------------------------------------------------ #
    @property
    def shape(self):
        """
        Shape of the object's internal matrix representation

        :return: (4,4)
        :rtype: tuple

        Each value within the ``SE3`` instance is a NumPy array of this shape.
        """
        return (4, 4)

    @property
    def t(self):
        """
        Translational component of SE(3)

        :return: translational component of SE(3)
        :rtype: numpy.ndarray

        ``x.t`` is the translational component of ``x`` as an array with
        shape (3,). If ``len(x) > 1``, return an array with shape=(N,3).

        .. runblock:: pycon

            >>> from spatialmath import UnitQuaternion

            >>> x = SE3(1,2,3)
            >>> x.t
            array([1., 2., 3.])
            >>> x = SE3([ SE3(1,2,3), SE3(4,5,6)])
            >>> x.t
            array([[1., 2., 3.],
                   [4., 5., 6.]])

        
        :SymPy: supported
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
        :rtype: SE3 instance

        Efficiently compute the inverse of each of the SE(3) values taking into
        account the matrix structure.

        .. math::
        
            T = \left[ \begin{array}{cc} \mat{R} & \vec{t} \\ 0 & 1 \end{array} \right],
            \mat{T}^{-1} = \left[ \begin{array}{cc} \mat{R}^T & -\mat{R}^T \vec{t} \\ 0 & 1 \end{array} \right]`

        Example::

            >>> x = SE3(1,2,3)
            >>> x.inv()
            SE3(array([[ 1.,  0.,  0., -1.],
                       [ 0.,  1.,  0., -2.],
                       [ 0.,  0.,  1., -3.],
                       [ 0.,  0.,  0.,  1.]]))

        :seealso: :func:`~spatialmath.base.transforms3d.trinv`

        :SymPy: supported
        """
        if len(self) == 1:
            return SE3(base.trinv(self.A), check=False)
        else:
            return SE3([base.trinv(x) for x in self.A], check=False)

    def delta(self, X2):
        r"""
        Infinitesimal difference of SE(3) values

        :return: differential motion vector
        :rtype: numpy.ndarray, shape=(6,)

        ``X1.delta(X2)`` is the differential motion (6x1) corresponding to
        infinitesimal motion (in the ``X1`` frame) from pose ``X1`` to ``X2``.

        The vector :math:`d = [\delta_x, \delta_y, \delta_z, \theta_x, \theta_y, \theta_z]`
        represents infinitesimal translation and rotation.

        Example::

            >>> x1 = SE3.Rx(0.3)
            >>> x2 = SE3.Rx(0.3001)
            >>> x1.delta(x2)
            array([0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 9.99999998e-05,
                0.00000000e+00, 0.00000000e+00])

        .. note::

            - the displacement is only an approximation to the motion, and assumes
              that ``X1`` ~ ``X2``.
            - can be considered as an approximation to the effect of spatial velocity over a
              a time interval, ie. the average spatial velocity multiplied by time.

        :Reference: Robotics, Vision & Control: Second Edition, P. Corke, Springer 2016; p67.

        :seealso: :func:`~spatialmath.base.transforms3d.tr2delta`
        """
        return base.tr2delta(self.A, X2.A)

    def Ad(self):
        """
        Adjoint of SE(3)

        :return: adjoint matrix
        :rtype: numpy.ndarray, shape=(6,6)

        ``SE3.Ad`` is the 6x6 adjoint matrix

        If spatial velocity :math:`\nu = (v_x, v_y, v_z, \omega_x, \omega_y, \omega_z)^T`
        and the SE(3) represents the pose of {B} relative to {A}, 
        ie. :math:`{}^A {\bf T}_B, and the adjoint is :math:`\mathbf{A}` then
        :math:`{}^{A}\!\nu = \mathbf{A} {}^{B}\!\nu`.

        .. warning:: Do not use this method to map velocities 
            between robot base and end-effector frames - use ``jacob()``.

        .. note:: Use this method to map velocities between two frames on
            the same rigid-body.  

        :reference: Robotics, Vision & Control: Second Edition, P. Corke, Springer 2016; p65.
        :seealso: SE3.jacob, Twist.ad, :func:`~spatialmath.base.tr2jac`
        :SymPy: supported
        """
        return base.tr2adjoint(self.A)

    def jacob(self):
        """
        Velocity transform for SE(3)

        :return: Jacobian matrix
        :rtype: numpy.ndarray, shape=(6,6)

        ``SE3.jacob()`` is the 6x6 Jacobian that maps spatial velocity or
        differential motion from frame {B} to frame {A} where the pose of {B}
        relative to {A} is represented by the homogeneous transform T =
        :math:`{}^A {\bf T}_B`.  
        
        .. note::
            - To map from frame {A} to frame {B} use the transpose of this matrix.
            - Use this method to map velocities between the robot end-effector frame
              and the base frames.

        .. warning:: Do not use this method to map velocities between two frames
            on the same rigid-body.

        :seealso: SE3.Ad, Twist.ad, :func:`~spatialmath.base.tr2jac`
        :Reference: Robotics, Vision & Control: Second Edition, P. Corke, Springer 2016; p65.
        :SymPy: supported
        """
        return base.tr2jac(self.A)

    def twist(self):
        """
        SE(3) as twist

        :return: equivalent rigid-body motion as a twist vector
        :rtype: Twist3 instance

        Example::

            >>> x = SE3(1,2,3)
            >>> x.twist()
            Twist3([1, 2, 3, 0, 0, 0])

        :seealso: :func:`spatialmath.twist.Twist3`
        """
        from spatialmath.twist import Twist3

        return Twist3(self.log(twist=True))
    # ------------------------------------------------------------------------ #

    @staticmethod
    def isvalid(x, check=True):
        """
        Test if matrix is a valid SE(3)

        :param x: matrix to test
        :type x: numpy.ndarray
        :return: ``True`` if the matrix is 4x4 and a valid element of SE(3), ie. it
                 is a valid homogeneous transformation matrix.
        :rtype: bool

        :seealso: :func:`~spatialmath.base.transforms3d.ishom`
        """
        return base.ishom(x, check=check)

    # ---------------- variant constructors ---------------------------------- #

    @classmethod
    def Rx(cls, theta, unit='rad', t=None):
        """
        Create anSE(3) pure rotation about the X-axis

        :param θ: rotation angle about X-axis
        :type θ: float
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :param t: translation, optional
        :type t: 3-element array-like
        :return: SE(3) matrix
        :rtype: SE3 instance

        - ``SE3.Rx(θ)`` is an SE(3) rotation of θ radians about the x-axis
        - ``SE3.Rx(θ, "deg")`` as above but θ is in degrees
        - ``SE3.Rx(θ, t=T)`` as above but also sets the translational component

        If ``θ`` is an array then the result is a sequence of rotations defined
        by consecutive elements.

        .. note:: The translation option only works for the scalar θ case.

        Example:

        .. runblock:: pycon

            >>> SE3.Rx(0.3)
            >>> SE3.Rx([0.3, 0.4])

        :seealso: :func:`~spatialmath.base.transforms3d.trotx`
        :SymPy: supported
        """
        return cls([base.trotx(x, t=t, unit=unit) for x in base.getvector(theta)], check=False)

    @classmethod
    def Ry(cls, theta, unit='rad', t=None):
        """
        Create an SE(3) pure rotation about the Y-axis

        :param θ: rotation angle about X-axis
        :type θ: float
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :param t: translation, optional
        :type t: 3-element array-like
        :return: SE(3) matrix
        :rtype: SE3 instance

        - ``SE3.Ry(θ)`` is an SO(3) rotation of θ radians about the y-axis
        - ``SE3.Ry(θ, "deg")`` as above but θ is in degrees
        - ``SE3.Ry(θ, t=T)`` as above but also sets the translational component

        If ``θ`` is an array then the result is a sequence of rotations defined
        by consecutive elements.

        .. note:: The translation option only works for the scalar θ case.

        Example:

        .. runblock:: pycon

            >>> SE3.Ry(0.3)
            >>> SE3.Ry([0.3, 0.4])

        :seealso: :func:`~spatialmath.base.transforms3d.troty`
        :SymPy: supported
        """
        return cls([base.troty(x, t=t, unit=unit) for x in base.getvector(theta)], check=False)

    @classmethod
    def Rz(cls, theta, unit='rad', t=None):
        """
        Create an SE(3) pure rotation about the Z-axis

        :param θ: rotation angle about Z-axis
        :type θ: float
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :param t: translation, optional
        :type t: 3-element array-like
        :return: SE(3) matrix
        :rtype: SE3 instance

        - ``SE3.Rz(θ)`` is an SO(3) rotation of θ radians about the z-axis
        - ``SE3.Rz(θ, "deg")`` as above but θ is in degrees
        - ``SE3.Rz(θ, t=T)`` as above but also sets the translational component

        If ``θ`` is an array then the result is a sequence of rotations defined
        by consecutive elements.

        .. note:: The translation option only works for the scalar θ case.

        Example:

        .. runblock:: pycon

            >>> SE3.Rz(0.3)
            >>> SE3.Rz([0.3, 0.4])

        :seealso: :func:`~spatialmath.base.transforms3d.trotz`
        :SymPy: supported
        """
        return cls([base.trotz(x, t=t, unit=unit) for x in base.getvector(theta)], check=False)

    @classmethod
    def Rand(cls, N=1, xrange=(-1, 1), yrange=(-1, 1), zrange=(-1, 1)):  # pylint: disable=arguments-differ
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
        :return: SE(3) matrix
        :rtype: SE3 instance

        Return an SE3 instance with random rotation and translation.

        - ``SE3.Rand()`` is a random SE(3) translation.
        - ``SE3.Rand(N)`` is an SE3 object containing a sequence of N random
          poses.

        Example::

            >>> SE3.Rand(2)
            SE3([
            array([[ 0.58076657,  0.64578702, -0.49565041, -0.78585825],
                [-0.57373134, -0.10724881, -0.8119914 ,  0.72069253],
                [-0.57753142,  0.75594763,  0.30822173,  0.12291999],
                [ 0.        ,  0.        ,  0.        ,  1.        ]]),
            array([[ 0.96481299, -0.26267256, -0.01179066,  0.80294729],
                [ 0.06421463,  0.19190584,  0.97931028, -0.15021311],
                [-0.25497525, -0.94560841,  0.20202067,  0.02684599],
                [ 0.        ,  0.        ,  0.        ,  1.        ]]) ])

        :seealso: :func:`~spatialmath.quaternions.UnitQuaternion.Rand`
        """
        X = np.random.uniform(low=xrange[0], high=xrange[1], size=N)  # random values in the range
        Y = np.random.uniform(low=yrange[0], high=yrange[1], size=N)  # random values in the range
        Z = np.random.uniform(low=yrange[0], high=zrange[1], size=N)  # random values in the range
        R = SO3.Rand(N=N)
        return cls([base.transl(x, y, z) @ base.r2t(r.A) for (x, y, z, r) in zip(X, Y, Z, R)], check=False)

    @classmethod
    def Eul(cls, *angles, unit='rad'):
        r"""
        Create an SE(3) pure rotation from Euler angles

        :param 𝚪: Euler angles
        :type 𝚪: array_like or numpy.ndarray with shape=(N,3)
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :return: SE(3) matrix
        :rtype: SE3 instance

        - ``SE3.Eul(𝚪)`` is an SE(3) rotation defined by a 3-vector of Euler
          angles :math:`\Gamma=(\phi, \theta, \psi)` which correspond to
          consecutive rotations about the Z, Y, Z axes respectively.

        If ``𝚪`` is an Nx3 matrix then the result is a sequence of
        rotations each defined by Euler angles corresponding to the rows of
        ``𝚪``.

        - ``SE3.Eul(φ, θ, ψ)`` as above but the angles are provided as three
          scalars.

        Example:

        .. runblock:: pycon
        
            >>> from spatialmath import SE3
            >>> SE3.Eul(0.1, 0.2, 0.3)
            >>> SE3.Eul([0.1, 0.2, 0.3])
            >>> SE3.Eul(10, 20, 30, unit='deg')

        :seealso: :func:`~spatialmath.pose3d.SE3.eul`, :func:`~spatialmath.base.transforms3d.eul2r`
        :SymPy: supported
        """
        if len(angles) == 1:
            angles = angles[0]
        if base.isvector(angles, 3):
            return cls(base.eul2tr(angles, unit=unit), check=False)
        else:
            return cls([base.eul2tr(a, unit=unit) for a in angles], check=False)

    @classmethod
    def RPY(cls, *angles, unit='rad', order='zyx'):
        r"""
        Create an SE(3) pure rotation from roll-pitch-yaw angles

        :param 𝚪: roll-pitch-yaw angles
        :type 𝚪: array_like or numpy.ndarray with shape=(N,3)
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :param order: rotation order: 'zyx' [default], 'xyz', or 'yxz'
        :type order: str
        :return: SE(3) matrix
        :rtype: SE3 instance

        - ``SE3.RPY(𝚪)`` is an SE(3) rotation defined by a 3-vector of roll,
          pitch, yaw angles :math:`\Gamma=(r, p, y)` which correspond to
          successive rotations about the axes specified by ``order``:

            - ``'zyx'`` [default], rotate by yaw about the z-axis, then by pitch about the new y-axis,
              then by roll about the new x-axis.  This is the **convention** for a mobile robot with x-axis forward
              and y-axis sideways.
            - ``'xyz'``, rotate by yaw about the x-axis, then by pitch about the new y-axis,
              then by roll about the new z-axis. This is the **convention** for a robot gripper with z-axis forward
              and y-axis between the gripper fingers.
            - ``'yxz'``, rotate by yaw about the y-axis, then by pitch about the new x-axis,
              then by roll about the new z-axis. This is the **convention** for a camera with z-axis parallel
              to the optical axis and x-axis parallel to the pixel rows.

        If ``𝚪`` is an Nx3 matrix then the result is a sequence of rotations each defined by RPY angles
        corresponding to the rows of ``𝚪``.

        - ``SE3.RPY(⍺, β, 𝛾)`` as above but the angles are provided as three
          scalars.

        Example:

        .. runblock:: pycon
        
            >>> from spatialmath import SE3
            >>> SE3.RPY(0.1, 0.2, 0.3)
            >>> SE3.RPY([0.1, 0.2, 0.3])
            >>> SE3.RPY(0.1, 0.2, 0.3, order='xyz')
            >>> SE3.RPY(10, 20, 30, unit='deg')

        :seealso: :func:`~spatialmath.pose3d.SE3.rpy`, :func:`~spatialmath.base.transforms3d.rpy2r`
        :SymPy: supported
        """
        if len(angles) == 1:
            angles = angles[0]

        if base.isvector(angles, 3):
            return cls(base.rpy2tr(angles, order=order, unit=unit), check=False)
        else:
            return cls([base.rpy2tr(a, order=order, unit=unit) for a in angles], check=False)

    @classmethod
    def OA(cls, o, a):
        r"""
        Create an SE(3) pure rotation from two vectors

        :param o: 3-vector parallel to Y- axis
        :type o: array_like
        :param a: 3-vector parallel to the Z-axis
        :type a: array_like
        :return: SE(3) matrix
        :rtype: SE3 instance

        ``SE3.OA(o, a)`` is an SE(3) rotation defined in terms of vectors ``o``
        and ``a`` respectively parallel to the Y- and Z-axes of its reference
        frame.  In robotics these axes are respectively called the *orientation*
        and *approach* vectors defined such that :math:`\mathbf{R} = [n, o, a]`
        and :math:`n = o \times a`.

        .. note::

            - The ``a`` vector is the only guaranteed to have the same direction in the resulting
              rotation matrix
            - ``o`` and ``a`` do not have to be unit-length, they are normalized
            - ``o`` and ``a`` do not have to be orthogonal, so long as they are not parallel
              ``o`` is adjusted to be orthogonal to ``a``.

        Example::

            >>> SE3.OA([1, 0, 0], [0, 0, -1])
            SE3(array([[-0.,  1.,  0.,  0.],
                    [ 1.,  0.,  0.,  0.],
                    [ 0.,  0., -1.,  0.],
                    [ 0.,  0.,  0.,  1.]]))

        :seealso: :func:`~spatialmath.base.transforms3d.oa2r`
        """
        return cls(base.oa2tr(o, a), check=False)

    @classmethod
    def AngVec(cls, theta, v, *, unit='rad'):
        r"""
        Create an SE(3) pure rotation matrix from rotation angle and axis

        :param θ: rotation
        :type θ: float
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :param v: rotation axis, 3-vector
        :type v: array_like
        :return: SE(3) matrix
        :rtype: SE3 instance

        ``SE3.AngVec(θ, v)`` is an SE(3) rotation defined by
        a rotation of ``θ`` about the vector ``v``.

        .. math::
        
            \mbox{if}\,\, \theta \left\{ \begin{array}{ll}
                = 0 & \mbox{return identity matrix}\\
                \ne 0 & \mbox{v must have a finite length}
                \end{array}
                \right.

        :seealso: :func:`~spatialmath.pose3d.SE3.angvec`, :func:`~spatialmath.pose3d.SE3.EulerVec`, :func:`~spatialmath.base.transforms3d.angvec2r`
        """
        return cls(base.angvec2tr(theta, v, unit=unit), check=False)

    @classmethod
    def EulerVec(cls, w):
        r"""
        Construct a new SE(3) pure rotation matrix from an Euler rotation vector

        :param ω: rotation axis
        :type ω: 3-element array_like
        :return: SE(3) rotation
        :rtype: SE3 instance

        ``SE3.EulerVec(ω)`` is a unit quaternion that describes the 3D rotation
        defined by a rotation of :math:`\theta = \lVert \omega \rVert` about the
        unit 3-vector :math:`\omega / \lVert \omega \rVert`.

        Example:

        .. runblock:: pycon
        
            >>> from spatialmath import SE3
            >>> SE3.EulerVec([0.5,0,0])

        .. note:: :math:`\theta = 0` the result in an identity matrix, otherwise
            ``V`` must have a finite length, ie. :math:`|V| > 0`.

        :seealso: :func:`~spatialmath.pose3d.SE3.AngVec`, :func:`~spatialmath.base.transforms3d.angvec2tr`
        """
        assert base.isvector(w, 3), 'w must be a 3-vector'
        w = base.getvector(w)
        theta = base.norm(w)
        return cls(base.angvec2tr(theta, w), check=False)

    @classmethod
    def Exp(cls, S, check=True):
        """
        Create an SE(3) matrix from se(3)

        :param S: Lie algebra se(3) matrix
        :type S: numpy ndarray
        :return: SE(3) matrix
        :rtype: SE3 instance

        - ``SE3.Exp(S)`` is an SE(3) rotation defined by its Lie algebra
          which is a 4x4 se(3) matrix (skew symmetric)
        - ``SE3.Exp(t)`` is an SE(3) rotation defined by a 6-element twist
          vector (the unique elements of the se(3) skew-symmetric matrix)

        :seealso: :func:`~spatialmath.base.transforms3d.trexp`, :func:`~spatialmath.base.transformsNd.skew`
        """
        if base.isvector(S, 6):
            return cls(base.trexp(base.getvector(S)), check=False)
        else:
            return cls(base.trexp(S), check=False)
            

    @classmethod
    def Delta(cls, d):
        r"""
        Create SE(3) from differential motion

        :param d: differential motion
        :type d: 6-element array_like
        :return: SE(3) matrix
        :rtype: SE3 instance


        ``T = delta2tr(d)`` is an SE(3) representing differential 
        motion :math:`d = [\delta_x, \delta_y, \delta_z, \theta_x, \theta_y, \theta_z]`.

        :Reference: Robotics, Vision & Control: Second Edition, P. Corke, Springer 2016; p67.

        :seealso: :func:`~delta`, :func:`~spatialmath.base.transform3d.delta2tr`
        :SymPy: supported
        """
        return cls(base.trnorm(base.delta2tr(d)))

    @classmethod
    def Tx(cls, x):
        """
        Create an SE(3) translation along the X-axis

        :param x: translation distance along the X-axis
        :type x: float
        :return: SE(3) matrix
        :rtype: SE3 instance

        `SE3.Tx(x)` is an SE(3) translation of ``x`` along the x-axis

        Example:

        .. runblock:: pycon

            >>> SE3.Tx(2)
            >>> SE3.Tx([2,3])


        :seealso: :func:`~spatialmath.base.transforms3d.transl`
        :SymPy: supported
        """
        return cls([base.transl(_x, 0, 0) for _x in base.getvector(x)], check=False)


    @classmethod
    def Ty(cls, y):
        """
        Create an SE(3) translation along the Y-axis

        :param y: translation distance along the Y-axis
        :type y: float
        :return: SE(3) matrix
        :rtype: SE3 instance

        `SE3.Ty(y) is an SE(3) translation of ``y`` along the y-axis

        Example:

        .. runblock:: pycon

            >>> SE3.Ty(2)
            >>> SE3.Tz([2,3])


        :seealso: :func:`~spatialmath.base.transforms3d.transl`
        :SymPy: supported
        """
        return cls([base.transl(0, _y, 0) for _y in base.getvector(y)], check=False)

    @classmethod
    def Tz(cls, z):
        """
        Create an SE(3) translation along the Z-axis

        :param z: translation distance along the Z-axis
        :type z: float
        :return: SE(3) matrix
        :rtype: SE3 instance

        `SE3.Tz(z)` is an SE(3) translation of ``z`` along the z-axis

        Example:

        .. runblock:: pycon

            >>> SE3.Tz(2)
            >>> SE3.Tz([2,3])

        :seealso: :func:`~spatialmath.base.transforms3d.transl`
        :SymPy: supported
        """
        return cls([base.transl(0, 0, _z) for _z in base.getvector(z)], check=False)

    @classmethod
    def Rt(cls, R, t, check=True):
        """
        Create an SE(3) from rotation and translation

        :param R: rotation
        :type R: SO3 or ndarray(3,3)
        :param t: translation
        :type t: array_like(3)
        :param check: check rotation validity, defaults to True
        :type check: bool, optional
        :raises ValueError: bad rotation matrix
        :return: SE(3) matrix
        :rtype: SE3 instance
        """
        if isinstance(R, SO3):
            R = R.A
        elif base.isrot(R, check=check):
            pass
        else:
            raise ValueError('expecting SO3 or rotation matrix')

        return cls(base.rt2tr(R, t))

    # @classmethod
    # def SO3(cls, R, t=None, check=True):
    #     if isinstance(R, SO3):
    #         R = R.A
    #     elif base.isrot(R, check=check):
    #         pass
    #     else:
    #         raise ValueError('expecting SO3 or rotation matrix')
    #     if t is None:
    #         return cls(base.r2t(R))
    #     else:
    #         return cls(base.rt2tr(R, t))

if __name__ == '__main__':   # pragma: no cover

    import pathlib

    a = SE3(1,2,3)
    b = SO3(a)
    print(b)

    a = SO3.RPY([.1, .2, .3])
    b = SE3(a)
    print(b)

    from spatialmath import SE2
    a = SE2(1,2,.4)
    b = SE3(a)
    print(b)


    exec(open(pathlib.Path(__file__).parent.parent.absolute() / "tests" / "test_pose3d.py").read())  # pylint: disable=exec-used
