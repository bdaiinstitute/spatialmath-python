#!/usr/bin/env python3

import numpy as np
import math
from collections import namedtuple
from collections import UserList

from  spatialmath.base import argcheck as arg
import spatialmath.base as base
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from spatialmath import SE3
from spatialmath.smuserlist import SMUserList

_eps = np.finfo(np.float64).eps

# ======================================================================== #

class Plane:
    r"""
    Create a plane object from linear coefficients
    
    :param c: Plane coefficients
    :type c: 4-element array_like
    :return: a Plane object
    :rtype: Plane

    Planes are represented by the 4-vector :math:`[a, b, c, d]` which describes
    the plane :math:`\pi: ax + by + cz + d=0`.
    """
    def __init__(self, c):

        self.plane = arg.getvector(c, 4)
    
    # point and normal
    @classmethod
    def PN(cls, p, n):
        """
        Create a plane object from point and normal
        
        :param p: Point in the plane
        :type p: 3-element array_like
        :param n: Normal to the plane
        :type n: 3-element array_like
        :return: a Plane object
        :rtype: Plane

        """
        n = arg.getvector(n, 3)  # normal to the plane
        p = arg.getvector(p, 3)  # point on the plane
        return cls(np.r_[n, -np.dot(n, p)])
    
    # point and normal
    @classmethod
    def P3(cls, p):
        """
        Create a plane object from three points
        
        :param p: Three points in the plane
        :type p: numpy.ndarray, shape=(3,3)
        :return: a Plane object
        :rtype: Plane
        """
        
        p = arg.ismatrix((3,3))
        v1 = p[:,0]
        v2 = p[:,1]
        v3 = p[:,2]
        
        # compute a normal
        n = np.cross(v2-v1, v3-v1)
        
        return cls(n, v1)
        
    # line and point
    # 3 points
        
    @property
    def n(self):
        r"""
        Normal to the plane
        
        :return: Normal to the plane
        :rtype: 3-element array_like
        
        For a plane :math:`\pi: ax + by + cz + d=0` this is the vector
        :math:`[a,b,c]`.

        """
        # normal
        return self.plane[:3]
    
    @property
    def d(self):
        r"""
        Plane offset
        
        :return: Offset of the plane
        :rtype: float
        
        For a plane :math:`\pi: ax + by + cz + d=0` this is the scalar
        :math:`d`.

        """
        return self.plane[3]
    
    def contains(self, p, tol=10*_eps):
        """
        
        :param p: A 3D point
        :type p: 3-element array_like
        :param tol: Tolerance, defaults to 10*_eps
        :type tol: float, optional
        :return: if the point is in the plane
        :rtype: bool

        """
        return abs(np.dot(self.n, p) - self.d) < tol
    
    def __str__(self):
        """
        
        :return: String representation of plane
        :rtype: str

        """
        return str(self.plane)

# ======================================================================== #

class Plucker(SMUserList):
    """
    Plucker coordinate class
    
    Concrete class to represent a 3D line using Plucker coordinates.
    
    Methods:
        
    Plucker            Contructor from points
    Plucker.planes     Constructor from planes
    Plucker.pointdir   Constructor from point and direction
    
    Information and test methods::
    closest            closest point on line
    commonperp         common perpendicular for two lines
    contains           test if point is on line
    distance           minimum distance between two lines
    intersects         intersection point for two lines
    intersect_plane    intersection points with a plane
    intersect_volume   intersection points with a volume
    pp                 principal point
    ppd                principal point distance from origin
    point              generate point on line
    
    Conversion methods::
    char               convert to human readable string
    double             convert to 6-vector
    skew               convert to 4x4 skew symmetric matrix
    
    Display and print methods::
    display            display in human readable form
    plot               plot line
    
    Operators:
    *                  multiply Plucker matrix by a general matrix
    |                  test if lines are parallel
    ^                  test if lines intersect
    ==                 test if two lines are equivalent
    ~=                 test if lines are not equivalent

    Notes:
        
     - This is reference (handle) class object
     - Plucker objects can be used in vectors and arrays
    
    References:
        
     - Ken Shoemake, "Ray Tracing News", Volume 11, Number 1
       http://www.realtimerendering.com/resources/RTNews/html/rtnv11n1.html#art3
     - Matt Mason lecture notes http://www.cs.cmu.edu/afs/cs/academic/class/16741-s07/www/lectures/lecture9.pdf
     - Robotics, Vision & Control: Second Edition, P. Corke, Springer 2016; p596-7.
    
    Implementation notes:
        
     - The internal representation is a 6-vector [v, w] where v (moment), w (direction).
     - There is a huge variety of notation used across the literature, as well as the ordering
       of the direction and moment components in the 6-vector.
    
    Copyright (C) 1993-2019 Peter I. Corke
    """

    # w  # direction vector
    # v  # moment vector (normal of plane containing line and origin)
    
    def __init__(self, v=None, w=None):
        """
        Create a Plucker 3D line object
        
        :param v: Plucker vector, Plucker object, Plucker moment
        :type v: 6-element array_like, Plucker instance, 3-element array_like
        :param w: Plucker direction, optional
        :type w: 3-element array_like, optional
        :raises ValueError: bad arguments
        :return: Plucker line
        :rtype: Plucker

        - ``L = Plucker(X)`` creates a Plucker object from the Plucker coordinate vector
          ``X`` = [V,W] where V (3-vector) is the moment and W (3-vector) is the line direction.

        - ``L = Plucker(L)`` creates a copy of the Plucker object ``L``.
        
        - ``L = Plucker(V, W)`` creates a Plucker object from moment ``V`` (3-vector) and
          line direction ``W`` (3-vector).
          
        Notes:
            
        - The Plucker object inherits from ``collections.UserList`` and has list-like
          behaviours.
        - A single Plucker object contains a 1D array of Plucker coordinates.
        - The elements of the array are guaranteed to be Plucker coordinates.
        - The number of elements is given by ``len(L)``
        - The elements can be accessed using index and slice notation, eg. ``L[1]`` or
          ``L[2:3]``
        - The Plucker instance can be used as an iterator in a for loop or list comprehension.
        - Some methods support operations on the internal list.
          
        :seealso: Plucker.PQ, Plucker.Planes, Plucker.PointDir
        """
        super().__init__()  # enable list powers

        if w is None:
            # zero or one arguments passed
            if super().arghandler(v, convertfrom=(SE3,)):
                return

        else:
            # additional arguments
            assert arg.isvector(v, 3) and arg.isvector(w, 3), 'expecting two 3-vectors'
            self.data = [np.r_[v, w]]
            
        # needed to allow __rmul__ to work if left multiplied by ndarray
        #self.__array_priority__ = 100  

    @property
    def shape(self):
        return (6,)

    @staticmethod
    def _identity():
        return np.zeros((6,))

    @staticmethod
    def isvalid(x, check=False):
        return x.shape == (6,)

    @staticmethod
    def PQ(P=None, Q=None):
        """
        Create Plucker line object from two 3D points
        
        :param P: First 3D point
        :type P: 3-element array_like
        :param Q: Second 3D point
        :type Q: 3-element array_like
        :return: Plucker line
        :rtype: Plucker

        ``L = Plucker(P, Q)`` create a Plucker object that represents
        the line joining the 3D points ``P`` (3-vector) and ``Q`` (3-vector). The direction
        is from ``Q`` to ``P``.

        :seealso: Plucker, Plucker.Planes, Plucker.PointDir
        """
        P = arg.getvector(P, 3)
        Q = arg.getvector(Q, 3)
        # compute direction and moment
        w = P - Q
        v = np.cross(P - Q, P)
        return Plucker(np.r_[v, w])
    
    @staticmethod
    def Planes(pi1, pi2):
        r"""
        Create Plucker line from two planes
                
        :param pi1: First plane
        :type pi1: 4-element array_like, or Plane
        :param pi2: Second plane
        :type pi2: 4-element array_like, or Plane
        :return: Plucker line
        :rtype: Plucker

        ``L = Plucker.planes(PI1, PI2)`` is a Plucker object that represents
        the line formed by the intersection of two planes ``PI1`` and ``PI2``.

        Planes are represented by the 4-vector :math:`[a, b, c, d]` which describes
        the plane :math:`\pi: ax + by + cz + d=0`.
           
        :seealso: Plucker, Plucker.PQ, Plucker.PointDir
        """

        if not isinstance(pi1, Plane):
            pi1 = Plane(arg.getvector(pi1, 4))
        if not isinstance(pi2, Plane):
            pi2 = Plane(arg.getvector(pi2, 4))
        
        w = np.cross(pi1.n, pi2.n)
        v = pi2.d * pi1.n - pi1.d * pi2.n
        return Plucker(np.r_[v, w])

    @staticmethod
    def PointDir(point, dir):
        """
        Create Plucker line from point and direction
        
        :param point: A 3D point
        :type point: 3-element array_like
        :param dir: Direction vector
        :type dir: 3-element array_like
        :return: Plucker line
        :rtype: Plucker
        
        ``L = Plucker.pointdir(P, W)`` is a Plucker object that represents the
        line containing the point ``P`` and parallel to the direction vector ``W``.

        :seealso: Plucker, Plucker.Planes, Plucker.PQ
        """

        point = arg.getvector(point, 3)
        dir = arg.getvector(dir, 3)
        
        return Plucker(np.r_[np.cross(dir, point), dir])
    
    def append(self, x):
        """
        
        :param x: Plucker object
        :type x: Plucker
        :raises ValueError: Attempt to append a non Plucker object
        :return: Plucker object with new Plucker line appended
        :rtype: Plucker

        """
        #print('in append method')
        if not type(self) == type(x):
            raise ValueError("can pnly append Plucker object")
        if len(x) > 1:
            raise ValueError("cant append a Plucker sequence - use extend")
        super().append(x.A)

    @property
    def A(self):
        # get the underlying numpy array
        if len(self.data) == 1:
            return self.data[0]
        else:
            return self.data

    def __getitem__(self, i):
        # print('getitem', i, 'class', self.__class__)
        return self.__class__(self.data[i])
    
    @property
    def v(self):
        """
        Moment vector
        
        :return: the moment vector
        :rtype: numpy.ndarray, shape=(3,)

        """
        return self.data[0][0:3]
    
    @property
    def w(self):
        """
        Direction vector
        
        :return: the direction vector
        :rtype: numpy.ndarray, shape=(3,)
        
        :seealso: Plucker.uw

        """
        return self.data[0][3:6]
    
    @property
    def uw(self):
        """
        Line direction as a unit vector
        
        :return: Line direction
        :rtype: numpy.ndarray, shape=(3,)

        ``line.uw`` is a unit-vector parallel to the line.
        """
        return base.unitvec(self.w)
    
    @property
    def vec(self):
        """
        Line as a Plucker coordinate vector
        
        :return: Coordinate vector
        :rtype: numpy.ndarray, shape=(6,)
        
        ``line.vec`` is the  Plucker coordinate vector ``X`` = [V,W] where V (3-vector)
        is the moment and W (3-vector) is the line direction.
        """
        return np.r_[self.v, self.w]
    
    @property
    def skew(self):
        r"""
        Line as a Plucker skew-matrix
        
        :return: Skew-symmetric matrix form of Plucker coordinates
        :rtype: numpy.ndarray, shape=(4,4)

        ``M = line.skew()`` is the Plucker matrix, a 4x4 skew-symmetric matrix
        representation of the line.

        Notes:
            
         - For two homogeneous points P and Q on the line, :math:`PQ^T-QP^T` is also skew
           symmetric.
         - The projection of Plucker line by a perspective camera is a homogeneous line (3x1)
           given by :math:`\vee C M C^T` where :math:`C \in \mathbf{R}^{3 \times 4}` is the camera matrix.
        """
        
        v = self.v
        w = self.w
        
        # the following matrix is at odds with H&Z pg. 72
        return np.array([
                [ 0,     v[2], -v[1], w[0]],
                [-v[2],  0 ,    v[0], w[1]],
                [ v[1], -v[0],  0,    w[2]],
                [-w[0], -w[1], -w[2], 0   ]
            ])
    
    @property
    def pp(self):
        """
        Principal point of the line

        ``line.pp`` is the point on the line that is closest to the origin.

        Notes:
            
         - Same as Plucker.point(0)

        :seealso: Plucker.ppd, Plucker.point
        """
        
        return np.cross(self.v, self.w) / np.dot(self.w, self.w)    
    @property
    def ppd(self):
        """
        Distance from principal point to the origin

        :return: Distance from principal point to the origin
        :rtype: float
        
        ``line.ppd`` is the distance from the principal point to the origin.
        This is the smallest distance of any point on the line
        to the origin.

        :seealso: Plucker.pp
        """
        return math.sqrt(np.dot(self.v, self.v) / np.dot(self.w, self.w) )

    def point(self, lam):
        r"""
        Generate point on line
       
        :param lam: Scalar distance from principal point
        :type lam: float
        :return: Distance from principal point to the origin
        :rtype: float

        ``line.point(LAMBDA)`` is a point on the line, where ``LAMBDA`` is the parametric
        distance along the line from the principal point of the line such
        that :math:`P = P_p + \lambda \hat{d}` and :math:`\hat{d}` is the line
        direction given by ``line.uw``.

        :seealso: Plucker.pp, Plucker.closest, Plucker.uw
        """
        lam = arg.getvector(lam, out='row')
        return self.pp.reshape((3,1)) + self.uw.reshape((3,1)) * lam

    # ------------------------------------------------------------------------- #
    #  TESTS ON PLUCKER OBJECTS
    # ------------------------------------------------------------------------- #

    def contains(self, x, tol=50*_eps):
        """
        Test if points are on the line
        
        :param x: 3D point
        :type x: 3-element array_like, or numpy.ndarray, shape=(3,N)
        :param tol: Tolerance, defaults to 50*_eps
        :type tol: float, optional
        :raises ValueError: Bad argument
        :return: Whether point is on the line
        :rtype: bool or numpy.ndarray(N) of bool

        ``line.contains(X)`` is true if the point ``X`` lies on the line defined by
        the Plucker object self.
        
        If ``X`` is an array with 3 rows, the test is performed on every column and
        an array of booleans is returned.
        """
        if arg.isvector(x, 3):
            x = arg.getvector(x)
            return np.linalg.norm( np.cross(x - self.pp, self.w) ) < tol
        elif arg.ismatrix(x, (3,None)):
            return [np.linalg.norm(np.cross(_ - self.pp, self.w)) < tol for _ in x.T]
        else:
            raise ValueError('bad argument')

    def __eq__(l1, l2):  # pylint: disable=no-self-argument
        """
        Test if two lines are equivalent
        
        :param l1: First line
        :type l1: Plucker
        :param l2: Second line
        :type l2: Plucker
        :return: Plucker
        :return: line equivalence
        :rtype: bool

        ``L1 == L2`` is true if the Plucker objects describe the same line in
        space.  Note that because of the over parameterization, lines can be
        equivalent even if their coordinate vectors are different.
        """
        return abs( 1 - np.dot(base.unitvec(l1.vec), base.unitvec(l2.vec))) < 10*_eps
    
    def __ne__(l1, l2):  # pylint: disable=no-self-argument
        """
        Test if two lines are not equivalent
        
        :param l1: First line
        :type l1: Plucker
        :param l2: Second line
        :type l2: Plucker
        :return: line inequivalence
        :rtype: bool

        ``L1 != L2`` is true if the Plucker objects describe different lines in
        space.  Note that because of the over parameterization, lines can be
        equivalent even if their coordinate vectors are different.
        """
        
        return not l1.__eq__(l2)
    
    def isparallel(l1, l2, tol=10*_eps):  # pylint: disable=no-self-argument
        """
        Test if lines are parallel
        
        :param l1: First line
        :type l1: Plucker
        :param l2: Second line
        :type l2: Plucker
        :return: lines are parallel
        :rtype: bool

        ``l1.isparallel(l2)`` is true if the two lines are parallel.
        
        ``l1 | l2`` as above but in binary operator form

        :seealso: Plucker.or, Plucker.intersects
        """
        
        return np.linalg.norm(np.cross(l1.w, l2.w) ) < tol

    
    def __or__(l1, l2):  # pylint: disable=no-self-argument
        """
        Test if lines are parallel as a binary operator
        
        :param l1: First line
        :type l1: Plucker
        :param l2: Second line
        :type l2: Plucker
        :return: lines are parallel
        :rtype: bool

        ``l1 | l2`` is an operator which is true if the two lines are parallel.

        :seealso: Plucker.isparallel, Plucker.__xor__
        """
        return l1.isparallel(l2)

    
    def __xor__(l1, l2):  # pylint: disable=no-self-argument
        
        """
        Test if lines intersect as a binary operator
        
        :param l1: First line
        :type l1: Plucker
        :param l2: Second line
        :type l2: Plucker
        :return: lines intersect
        :rtype: bool

        ``l1 ^ l2`` is an operator which is true if the two lines intersect at a point.

        Notes:
            
         - Is false if the lines are equivalent since they would intersect at
           an infinite number of points.

        :seealso: Plucker.intersects, Plucker.parallel
        """
        return not l1.isparallel(l2) and (abs(l1 * l2) < 10*_eps )
    
    # ------------------------------------------------------------------------- #
    #  PLUCKER LINE DISTANCE AND INTERSECTION
    # ------------------------------------------------------------------------- #       
   
            
    def intersects(l1, l2):  # pylint: disable=no-self-argument
        """
        Intersection point of two lines
        
        :param l1: First line
        :type l1: Plucker
        :param l2: Second line
        :type l2: Plucker
        :return: 3D intersection point
        :rtype: numpy.ndarray, shape=(3,) or None

        ``l1.intersects(l2)`` is the point of intersection of the two lines, or
        ``None`` if the lines do not intersect or are equivalent.


        :seealso: Plucker.commonperp, Plucker.eq, Plucker.__xor__
        """
        if l1^l2:
            # lines do intersect
            return -(np.dot(l1.v, l2.w) * np.eye(3, 3) + \
                  l1.w.reshape((3,1)) @ l2.v.reshape((1,3)) - \
                  l2.w.reshape((3,1)) @ l1.v.reshape((1,3))) * base.unitvec(np.cross(l1.w, l2.w))
        else:
            # lines don't intersect
            return None
    
    def distance(l1, l2):  # pylint: disable=no-self-argument
        """
        Minimum distance between lines
        
        :param l1: First line
        :type l1: Plucker
        :param l2: Second line
        :type l2: Plucker
        :return: Closest distance
        :rtype: float

        ``l1.distance(l2) is the minimum distance between two lines.
        
        Notes:
            
         - Works for parallel, skew and intersecting lines.
         """
        if l1 | l2:
            # lines are parallel
            l = np.cross(l1.w, l1.v - l2.v * np.dot(l1.w, l2.w) / dot(l2.w, l2.w)) / np.linalg.norm(l1.w)
        else:
            # lines are not parallel
            if abs(l1 * l2) < 10*_eps:
                # lines intersect at a point
                l = 0
            else:
                # lines don't intersect, find closest distance
                l = abs(l1 * l2) / np.linalg.norm(np.cross(l1.w, l2.w))**2
        return l

    
    def closest(self, x):
        """
        Point on line closest to given point
        
        :param line: A line
        :type l1: Plucker
        :param l2: An arbitrary 3D point
        :type l2: 3-element array_like
        :return: Point on the line and distance to line
        :rtype: collections.namedtuple

        - ``line.closest(x).p`` is the coordinate of a point on the line that is
          closest to ``x``.

        - ``line.closest(x).d`` is the distance between the point on the line and ``x``.
        
        The return value is a named tuple with elements:
            
            - ``.p`` for the point on the line as a numpy.ndarray, shape=(3,)
            - ``.d`` for the distance to the point from ``x``
            - ``.lam`` the `lambda` value for the point on the line.

        :seealso: Plucker.point
        """
        # http://www.ahinson.com/algorithms_general/Sections/Geometry/PluckerLine.pdf
        # has different equation for moment, the negative

        x = arg.getvector(x, 3)

        lam = np.dot(x - self.pp, self.uw)
        p = self.point(lam).flatten()  # is the closest point on the line
        d = np.linalg.norm( x - p)
        
        return namedtuple('closest', 'p d lam')(p, d, lam)
    
    
    def commonperp(l1, l2):  # pylint: disable=no-self-argument
        """
        Common perpendicular to two lines
        
        :param l1: First line
        :type l1: Plucker
        :param l2: Second line
        :type l2: Plucker
        :return: Perpendicular line
        :rtype: Plucker or None

        ``l1.commonperp(l2)`` is the common perpendicular line between the two lines.
        Returns ``None`` if the lines are parallel.

        :seealso: Plucker.intersect
        """
        
        if l1 | l2:
            # no common perpendicular if lines are parallel
            return None
        else:
            # lines are skew or intersecting
            w = np.cross(l1.w, l2.w)
            v = np.cross(l1.v, l2.w) - np.cross(l2.v, l1.w) + \
                (l1 * l2) * np.dot(l1.w, l2.w) * base.unitvec(np.cross(l1.w, l2.w))
            
        return Plucker(v, w)


    def __mul__(left, right):  # pylint: disable=no-self-argument
        r"""
        Reciprocal product
        
        :param left: Left operand
        :type left: Plucker
        :param right: Right operand
        :type right: Plucker
        :return: reciprocal product
        :rtype: float

        ``left * right`` is the scalar reciprocal product :math:`\hat{w}_L \dot m_R + \hat{w}_R \dot m_R`.

        Notes:
            
         - Multiplication or composition of Plucker lines is not defined.
         - Pre-multiplication by an SE3 object is supported, see ``__rmul__``.

        :seealso: Plucker.__rmul__
        """
        if isinstance(right, Plucker):
            # reciprocal product
            return np.dot(left.uw, right.v) + np.dot(right.uw, left.v)
        else:
            raise ValueError('bad arguments')
        
    def __rmul__(right, left):  # pylint: disable=no-self-argument
        """
        Line transformation

        :param left: Rigid-body transform
        :type left: SE3
        :param right: Right operand
        :type right: Plucker
        :return: transformed line
        :rtype: Plucker
        
        ``T * line`` is the line transformed by the rigid body transformation ``T``.


        :seealso: Plucker.__mul__
        """
        if isinstance(left, SE3):
            A = np.r_[ np.c_[left.R,          base.skew(-left.t) @ left.R],
                       np.c_[np.zeros((3,3)), left.R]
                        ]
            return Plucker( A @ right.vec)  # premultiply by SE3
        else:
            raise ValueError('bad arguments')

    # ------------------------------------------------------------------------- #
    #  PLUCKER LINE DISTANCE AND INTERSECTION
    # ------------------------------------------------------------------------- #       


    def intersect_plane(line, plane):  # pylint: disable=no-self-argument
        r"""
        Line intersection with a plane
        
        :param line: A line
        :type line: Plucker
        :param plane: A plane
        :type plane: 4-element array_like or Plane
        :return: Intersection point
        :rtype: collections.namedtuple

        - ``line.intersect_plane(plane).p`` is the point where the line 
          intersects the plane, or None if no intersection.
         
        - ``line.intersect_plane(plane).lam`` is the `lambda` value for the point on the line
          that intersects the plane.

        The plane can be specified as:
            
         - a 4-vector :math:`[a, b, c, d]` which describes the plane :math:`\pi: ax + by + cz + d=0`.
         - a ``Plane`` object
         
         The return value is a named tuple with elements:
            
            - ``.p`` for the point on the line as a numpy.ndarray, shape=(3,)
            - ``.lam`` the `lambda` value for the point on the line.

        See also Plucker.point.
        """
        
        # Line U, V
        # Plane N n
        # (VxN-nU:U.N)
        # Note that this is in homogeneous coordinates.
        #    intersection of plane (n,p) with the line (v,p)
        #    returns point and line parameter
        
        if not isinstance(plane, Plane):
            plane = Plane(arg.getvector(plane, 4))
            
        den = np.dot(line.w, plane.n)
        
        if abs(den) > (100*_eps):
            # P = -(np.cross(line.v, plane.n) + plane.d * line.w) / den
            p = (np.cross(line.v, plane.n) - plane.d * line.w) / den
            
            t = np.dot( line.pp - p, plane.n)
            return namedtuple('intersect_plane', 'p lam')(p, t)
        else:
            return None

    def intersect_volume(self, bounds):
        """
        Line intersection with a volume
        
        :param line: A line
        :type line: Plucker
        :param bounds: Bounds of an axis-aligned rectangular cuboid
        :type plane: 6-element array_like
        :return: Intersection point
        :rtype: collections.namedtuple
        
        ``line.intersect_volume(bounds).p`` is a matrix (3xN) with columns
        that indicate where the line intersects the faces of the volume
        specified by ``bounds`` = [xmin xmax ymin ymax zmin zmax].  The number of
        columns N is either:
            
        - 0, when the line is outside the plot volume or,
        - 2 when the line pierces the bounding volume.
        
        ``line.intersect_volume(bounds).lam`` is an array of shape=(N,) where
        N is as above.
            
        The return value is a named tuple with elements:
            
            - ``.p`` for the points on the line as a numpy.ndarray, shape=(3,N)
            - ``.lam`` for the `lambda` values for the intersection points as a
              numpy.ndarray, shape=(N,).
        
        See also Plucker.plot, Plucker.point.
        """
        
        intersections = []
        
        # reshape, top row is minimum, bottom row is maximum
        bounds23 = bounds.reshape((3, 2))
        
        for face in range(0, 6):
            # for each face of the bounding volume
            #  x=xmin, x=xmax, y=ymin, y=ymax, z=zmin, z=zmax

            # planes are:
            #  0 normal in x direction, xmin
            #  1 normal in x direction, xmax
            #  2 normal in y direction, ymin
            #  3 normal in y direction, ymax
            #  4 normal in z direction, zmin
            #  5 normal in z direction, zmax
            
            i = face // 2  # 0, 1, 2
            I = np.eye(3,3)
            p = [0, 0, 0]
            p[i] = bounds[face]
            plane = Plane.PN(n=I[:,i], p=p)
            
            # find where line pierces the plane
            try:
                p, lam = self.intersect_plane(plane)
            except TypeError:
                continue  # no intersection with this plane
            
            # print('face %d: n=(%f, %f, %f)' % (face, plane.n[0], plane.n[1], plane.n[2]))
            # print('       : p=(%f, %f, %f)  ' % (p[0], p[1], p[2]))
            
            # print('face', face, ' point ', p, ' plane ', plane)
            # print('lamda', lam, self.point(lam))
            # find if intersection point is within the cube face
            #  test x,y,z simultaneously
            k = (p >= bounds23[:,0]) & (p <= bounds23[:,1])
            k = np.delete(k, i)  # remove the boolean corresponding to current face
            if all(k):
                # if within bounds, add
                intersections.append(lam)
                
#                     print('  HIT');

        # put them in ascending order
        intersections.sort()
        p = self.point(intersections)
        
        return namedtuple('intersect_volume', 'p lam')(p, intersections)

    
    # ------------------------------------------------------------------------- #
    #  PLOT AND DISPLAY
    # ------------------------------------------------------------------------- #   
    
    def plot(self, *pos, bounds=None, axis=None, **kwargs):
        """
         Plot a line
         
        :param line: A line
        :type line: Plucker
        :param bounds: Bounds of an axis-aligned rectangular cuboid as [xmin xmax ymin ymax zmin zmax], optional
        :type plane: 6-element array_like
        :param **kwargs: Extra arguents passed to `Line2D <https://matplotlib.org/3.2.2/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D>`_
        :return: Plotted line
        :rtype: Line3D or None

        - ``line.plot(bounds)`` adds a line segment to the current axes, and the handle of the line is returned.  
          The line segment is defined by the intersection of the line and the given rectangular cuboid. 
          If the line does not intersect the plotting volume None is returned.
          
        - ``line.plot()`` as above but the bounds are taken from the axis limits of the current axes.
          
        The line color or style is specified by:
        
            - a  MATLAB-style linestyle like 'k--'
            - additional arguments passed to `Line2D <https://matplotlib.org/3.2.2/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D>`_
            
        :seealso: Plucker.intersect_volume
        """
        if axis is None:
            ax = plt.gca()
        else:
            ax = axis

        if bounds is None:
            bounds = np.r_[ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]
        else:
            bounds = base.getvector(bounds, 6)
            ax.set_xlim(bounds[:2])
            ax.set_ylim(bounds[2:4])
            ax.set_zlim(bounds[4:6])

        # print(bounds)
        
        #U = self.Q - self.P;
        #line.p = self.P; line.v = unit(U);
        
        lines = []
        for line in self:
            P, lam = line.intersect_volume(bounds)
            
            if len(lam) > 0:
                l = ax.plot3D(P[0,:], P[1,:], P[2,:], *pos, **kwargs)
                lines.append(l)
        return lines

    def __str__(self):
        """
        Convert to a string
        
        :return: String representation of line parameters
        :rtype: str

        ``str(line)`` is a string showing Plucker parameters in a compact single
        line format like::
            
            { 0 0 0; -1 -2 -3}
            
        where the first three numbers are the moment, and the last three are the 
        direction vector.

        """
        
        return '\n'.join(['{{ {:.5g} {:.5g} {:.5g}; {:.5g} {:.5g} {:.5g}}}'.format(*list(base.removesmall(x.vec))) for x in self])

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
            return "Plucker([{:.5g}, {:.5g}, {:.5g}, {:.5g}, {:.5g}, {:.5g}])".format(*list(self.A))
        else:
            return "Plucker([\n" + \
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
        if len(self) == 1:
            p.begin_group(8, 'Plücker ')
            p.text(str(self))
            p.end_group(8, '')
        else:
            p.begin_group(8, 'Plücker(')
            for i, x in enumerate(self):
                p.break_()
                p.text(str(x))
            p.end_group(8, ')')

#         function z = side(self1, pl2)
#             Plucker.side Plucker side operator
# 
#             # X = SIDE(P1, P2) is the side operator which is zero whenever
#             # the lines P1 and P2 intersect or are parallel.
# 
#             # See also Plucker.or.
#             
#             if ~isa(self2, 'Plucker')
#                 error('SMTB:Plucker:badarg', 'both arguments to | must be Plucker objects');
#             end
#             L1 = pl1.line(); L2 = pl2.line();
#             
#             z = L1([1 5 2 6 3 4]) * L2([5 1 6 2 4 3])';
#         end

#         
#         function z = intersect(self1, pl2)
#             Plucker.intersect  Line intersection
#             
#             PL1.intersect(self2) is zero if the lines intersect.  It is positive if PL2
#             passes counterclockwise and negative if PL2 passes clockwise.  Defined as
#             looking in direction of PL1
#             
#                                        ---------->
#                            o                o
#                       ---------->
#                      counterclockwise    clockwise
#             
#             z = dot(self1.w, pl1.v) + dot(self2.w, pl2.v);
#         end
        
    # Static factory methods for constructors from exotic representations


    
if __name__ == '__main__':   # pragma: no cover

    import pathlib
    import os.path
    
    a = SE3.Exp([2,0,0,0,0,0])

    exec(open(os.path.join(pathlib.Path(__file__).parent.absolute(), "test_geom3d.py")).read())