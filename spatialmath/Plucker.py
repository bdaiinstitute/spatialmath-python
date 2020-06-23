#!/usr/bin/env python3
"""
Plucker Plucker coordinate class

Concrete class to represent a 3D line using Plucker coordinates.

Methods::
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

Operators::
*                  multiply Plucker matrix by a general matrix
|                  test if lines are parallel
^                  test if lines intersect
==                 test if two lines are equivalent
~=                 test if lines are not equivalent
#
Notes::
 - This is reference (handle) class object
 - Plucker objects can be used in vectors and arrays

References::
 - Ken Shoemake, "Ray Tracing News", Volume 11, Number 1
   http://www.realtimerendering.com/resources/RTNews/html/rtnv11n1.html#art3
 - Matt Mason lecture notes http://www.cs.cmu.edu/afs/cs/academic/class/16741-s07/www/lectures/lecture9.pdf
 - Robotics, Vision & Control: Second Edition, P. Corke, Springer 2016; p596-7.

Implementation notes::
 - The internal representation is two 3-vectors: v (direction), w (moment).
 - There is a huge variety of notation used across the literature, as well as the ordering
   of the direction and moment components in the 6-vector.


Copyright (C) 1993-2019 Peter I. Corke
"""

import numpy as np
import math
from collections import namedtuple
import spatialmath.base.argcheck as arg
import spatialmath.base as sm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from spatialmath import SE3

_eps = np.finfo(np.float64).eps

# NOTES
# working: constructor, origin-distance, plane+volume intersect, plot, .L
# method
# TODO
# .L method to skew

   
class Plane:
    
    def __init__(self, c=None, n=None, p=None):
        
        if c is not None:
            self.plane = arg.getvector(c, 4)
        elif n is not None and p is not None:
            n = arg.getvector(n, 3)  # normal to the plane
            p = arg.getvector(p, 3)  # point on the plane
            self.plane = np.r_[n, -np.dot(n, p)]
    
    @property
    def n(self):
        # normal
        return self.plane[:3]
    
    @property
    def d(self):
        return self.plane[3]
    
    def contains(self, p, tol=10*_eps):
        return abs(np.dot(self.n, p) - self.d) < tol

class Plucker:
    

    # w  # direction vector
    # v  # moment vector (normal of plane containing line and origin)
    
    def __init__(self, pl=None):
        """
        Plucker.Plucker Create Plucker line object

        P = Plucker(P1, P2) create a Plucker object that represents
        the line joining the 3D points P1 (3x1) and P2 (3x1). The direction
        is from P2 to P1.

        P = Plucker(X) creates a Plucker object from X (6x1) = [V,W] where
        V (3x1) is the moment and W (3x1) is the line direction.

        P = Plucker(L) creates a copy of the Plucker object L.

        Notes:
            
        - Planes are given by the 4-vector [a b c d] to represent ax+by+cz+d=0.\
        """

        if isinstance(pl, Plucker):
            self.v = pl.v
            self.w = pl.w
        elif arg.isvector(pl, 6):
            pl = arg.getvector(pl)
            self.v = pl[0:3]
            self.w = pl[3:6]
        else:
            raise ValueError('bad argument')


    @staticmethod
    def PQ(P=None, Q=None):
        P = arg.getvector(P, 3)
        Q = arg.getvector(Q, 3)
        # compute direction and moment
        w = P - Q
        v = np.cross(P - Q, P)
        return Plucker(np.r_[v, w])
    
    @staticmethod
    def VW(v=None, w=None):
        self.v = arg.getvector(v, 3)
        self.w = arg.getvector(w, 3)
        return Plucker(np.r_[v, w])
    
    @staticmethod
    def Planes(pi1, pi2):
        """
        Plucker.planes Create Plucker line from two planes

        P = Plucker.planes(PI1, PI2) is a Plucker object that represents
        the line formed by the intersection of two planes PI1, PI2 (each 4x1).

        Notes::
         - Planes are given by the 4-vector [a b c d] to represent ax+by+cz+d=0.
        """

        if not isinstance(p11, Plane):
            pi1 = Plane(arg.getvector(pi1, 4))
        if not isinstance(p12, Plane):
            pi2 = Plane(arg.getvector(pi2, 4))
        
        w = np.cross(pi1.n, pi2.n)
        v = pi2.d * pi1.n - pi1.d * pi2.n
        return Plucker(np.r_[v, w])

    @staticmethod
    def PointDir(point, dir):
        """
        Plucker.pointdir Construct Plucker line from point and direction

        P = Plucker.pointdir(P, W) is a Plucker object that represents the
        line containing the point P (3x1) and parallel to the direction vector W (3x1).

        See also: Plucker.
        """

        point = arg.getvector(point, 3)
        dir = arg.getvector(dir, 3)
        #                     self.P = B;
        #                     self.Q = A+B;
        
        return Plucker(np.r_[np.cross(dir, point), dir])
  
    @property
    def pp(self):
        """
        Plucker.pp Principal point of the line

        ``P = line.pp()`` is the point on the line that is closest to the origin.

        Notes:
            
         - Same as Plucker.point(0)

        See also Plucker.ppd, Plucker.point.
        """
        
        return np.cross(self.v, self.w) / np.dot(self.w, self.w)

    
    @property
    def uw(self):
        """
        Plucker.uw Line direction as a unit vector

        self.UW is a unit-vector parallel to the line
        """
        return sm.unitvec(self.w)
    
    @property
    def vec(self):
        return np.r_[self.v, self.w]
    
    @property
    def skew(self):
        """
        Plucker.skew Skew matrix form of the line

        ``L = line.skew()`` is the Plucker matrix, a 4x4 skew-symmetric matrix
        representation of the line.

        Notes:
            
         - For two homogeneous points P and Q on the line, PQ'-QP' is also skew
           symmetric.
         - The projection of Plucker line by a perspective camera is a homogeneous line (3x1)
           given by vex(C*L*C') where C (3x4) is the camera matrix.
        """
        
        v = self.v; w = self.w;
        
        # the following matrix is at odds with H&Z pg. 72
        return np.array([
                [ 0,     v[2], -v[1], w[0]],
                [-v[2],  0 ,    v[0], w[1]],
                [ v[1], -v[0],  0,    w[2]],
                [-w[0], -w[1], -w[2], 0   ]
            ])
    

    @property
    def ppd(self):
        """
        Plucker.ppd  Distance from principal point to the origin

        ``P = line.ppd()`` is the distance from the principal point to the origin.
        This is the smallest distance of any point on the line
        to the origin.

        See also Plucker.pp.
        """
        return math.sqrt(np.dot(self.v, self.v) / np.dot(self.w, self.w) )

       
    def point(L, lam):
        """
        Plucker.point Generate point on line

        ``P = self.point(LAMBDA)`` is a point on the line, where LAMBDA is the parametric
        distance along the line from the principal point of the line P = PP + self.UW*LAMBDA.

        See also Plucker.pp, Plucker.closest.
        """
        lam = arg.getvector(lam, out='row')
        return L.pp.reshape((3,1)) + L.uw.reshape((3,1)) * lam

    # ------------------------------------------------------------------------- #
    #  TESTS ON PLUCKER OBJECTS
    # ------------------------------------------------------------------------- #

    def contains(self, x, tol=50*_eps):
        """
        Plucker.contains  Test if point is on the line

        ``line.contains(X)`` is true if the point X (3x1) lies on the line defined by
        the Plucker object self.
        """
        if arg.isvector(x, 3):
            return np.linalg.norm( np.cross(x - self.pp, self.w) ) < tol
        elif arg.ismatrix(x, (3,None)):
            return [np.linalg.norm(np.cross(_ - self.pp, self.w)) < tol for _ in x.T]
        else:
            raise ValueError('bad argument')

    
    def __eq__(self, line):
        """
        Plucker.eq Test if two lines are equivalent

        PL1 == PL2 is true if the Plucker objects describe the same line in
        space.  Note that because of the over parameterization, lines can be
        equivalent even if they have different parameters.
        """
        
        return abs( 1 - np.dot(sm.unitvec(self.vec), sm.unitvec(line.vec))) < 10*_eps
    
    def __ne__(self, line):
        """
        Plucker.ne Test if two lines are not equivalent

        PL1 ~= PL2 is true if the Plucker objects describe different lines in
        space.  Note that because of the over parameterization, lines can be
        equivalent even if they have different parameters.
        """
        
        return not self.__eq__(line)
    
    def isparallel(p1, p2, tol=10*_eps):
        """
        Plucker.isparallel Test if lines are parallel

        P1.isparallel(P2) is true if the lines represented by Plucker objects P1
        and P2 are parallel.

        See also Plucker.or, Plucker.intersects.
        """
        
        return np.linalg.norm(np.cross(p1.w, p2.w) ) < tol

    
    def __or__(p1, p2):
        """
        Plucker.or Test if lines are parallel

        P1|P2 is true if the lines represented by Plucker objects P1
        and P2 are parallel.

        Notes::
         - Can be used in operator form as P1|P2.

        See also Plucker.isparallel, Plucker.mpower.
        """
        return p1.isparallel(p2)

    
    def __xor__(p1, p2):
        """
        Plucker.mpower Test if lines intersect

        P1^P2 is true if lines represented by Plucker objects P1
        and P2 intersect at a point.

        Notes::
         - Is false if the lines are equivalent since they would intersect at
           an infinite number of points.

        See also Plucker.intersects, Plucker.parallel.
        """
        return not isparallel(p1, p2) and (abs(p1 * p2) < 10*_eps )
    
    # ------------------------------------------------------------------------- #
    #  PLUCKER LINE DISTANCE AND INTERSECTION
    # ------------------------------------------------------------------------- #       
   
            
    def intersects(p1, p2):
        """
        Plucker.intersects Find intersection of two lines

        P = P1.intersects(P2) is the point of intersection (3x1) of the lines
        represented by Plucker objects P1 and P2.  P = [] if the lines
        do not intersect, or the lines are equivalent.

        Notes::
         - Can be used in operator form as P1^P2.
         - Returns [] if the lines are equivalent (P1==P2) since they would intersect at
           an infinite number of points.

        See also Plucker.commonperp, Plucker.eq, Plucker.mpower.
        """
        if p1^p2:
            return -(np.dot(p1.v, p2.w) * np.eye(3, 3) + \
                  p1.w.reshape((3,1)) @ p2.v.reshape((1,3)) - 
                  p2.w.reshape((3,1)) @ p1.v.reshape((1,3))) * sm.unitvec(np.cross(p1.w, p2.w))
        else:
            return None
    
    def distance(p1, p2):
        """
        Plucker.distance Distance between lines

        d = P1.distance(P2) is the minimum distance between two lines represented
        by Plucker objects P1 and P2.

        Notes::
         - Works for parallel, skew and intersecting lines.
         """
        if isparallel(p1, p2):
            # lines are parallel
            l = np.cross(p1.w, p1.v - p2.v * np.dot(p1.w, p2.w) / dot(p2.w, p2.w)) / np.linalg.norm(p1.w)
        else:
            # lines are not parallel
            if abs(p1 * p2) < 10*_eps:
                # lines intersect at a point
                l = 0
            else:
                # lines don't intersect, find closest distance
                l = abs(p1 * p2) / np.linalg.norm(np.cross(p1.w, p2.w))**2
        return l

    
    def closest(self, x):
        """
        Plucker.closest  Point on line closest to given point

        P = self.closest(X) is the coordinate of a point (3x1) on the line that is
        closest to the point X (3x1).

        [P,d] = self.closest(X) as above but also returns the minimum distance
        between the point and the line.

        [P,dist,lambda] = self.closest(X) as above but also returns the line parameter
        lambda corresponding to the point on the line, ie. P = self.point(lambda)

        See also Plucker.point.            
        """
        # http://www.ahinson.com/algorithms_general/Sections/Geometry/PluckerLine.pdf
        # has different equation for moment, the negative

        x = arg.getvector(x, 3)

        lam = np.dot(x - self.pp, self.uw)
        p = self.point(lam)  # is the closest point on the line
        d = np.linalg.norm( x - p)
        
        return namedtuple('closest', 'p d lam')(p, d, lam)
    
    
    def commonperp(p1, p2):
        """
        Plucker.commonperp Common perpendicular to two lines

        P = PL1.commonperp(self2) is a Plucker object representing the common
        perpendicular line between the lines represented by the Plucker objects
        PL1 and PL2.

        See also Plucker.intersect.
        """
        
        if isparallel(p1, p2):
            # no common perpendicular if lines are parallel
            return None
        else:
            w = np.cross(p1.w, p2.w)
            v = np.cross(p1.v, p2.w) - np.cross(p2.v, p1.w) + \
                (p1 * p2) * np.dot(p1.w, p2.w) * sm.unitvec(np.cross(p1.w, p2.w))
            
        return Plucker(np.r_[v, w])


    def __mul__(left, right):
        """
        Plucker.mtimes Plucker multiplication

        PL1 * PL2 is the scalar reciprocal product.

        PL * M is the product of the Plucker skew matrix and M (4xN).

        Notes::
         - The * operator is overloaded for convenience.
         - Multiplication or composition of Plucker lines is not defined.
         - Premultiplying by an SE3 will transform the line with respect to the world
           coordinate frame.

        See also Plucker.skew, SE3.mtimes.
        """
        
        if isinstance(left, Plucker) and isinstance(right, Plucker):
            # reciprocal product
            return np.dot(left.uw, right.v) + np.dot(right.uw, left.v)
        elif isinstance(left, Plucker) and arg.ismatrix(right, (4,None)):
            return  left.skew @ right;  # postmultiply by 4xN
        

    def __rmul__(right, left):
        """
        Plucker.mtimes Plucker multiplication

        M * PL is the product of M (Nx4) and the Plucker skew matrix (4x4).

        Notes::
         - The * operator is overloaded for convenience.
         - Multiplication or composition of Plucker lines is not defined.
         - Premultiplying by an SE3 will transform the line with respect to the world
           coordinate frame.

        See also Plucker.skew, SE3.mtimes.
        """
        
        if arg.ismatrix(left, (None,4)):
            return left @ right.skew  # premultiply by Nx4
        elif isinstance(left, SE3):
            return left.A @ right.skew  # premultiply by 4x4


    # ------------------------------------------------------------------------- #
    #  PLUCKER LINE DISTANCE AND INTERSECTION
    # ------------------------------------------------------------------------- #       


    def intersect_plane(L, plane):
        """
        Plucker.intersect_plane  Line intersection with plane

        X = self.intersect_plane(PI) is the point where the Plucker line PL 
        intersects the plane PI.  X=[] if no intersection.

        The plane PI can be either:
         - a vector (1x4) = [a b c d] to describe the plane ax+by+cz+d=0.
         - a structure with a normal PI.n (3x1) and an offset PI.p
           (1x1) such that PI.n X + PI.p = 0.  

        [X,lambda] = self.intersect_plane(P) as above but also returns the
        line parameter at the intersection point, ie. X = self.point(lambda).

        See also Plucker.point.
        """
        
        # Line U, V
        # Plane N n
        # (VxN-nU:U.N)
        # Note that this is in homogeneous coordinates.
        #    intersection of plane (n,p) with the line (v,p)
        #    returns point and line parameter
        
        
        den = np.dot(L.w, plane.n)
        
        if abs(den) > (100*_eps):
            P = -(np.cross(L.v, plane.n) + plane.p * L.w) / den
            p = (np.cross(L.v, plane.n) - plane.p * L.w) / den
            
            P = L.pp
            t = np.dot( P-p, N)
            return namedtuple('intersect_plane', 'p t')(P, t)
        else:
            return None

    def intersect_volume(line, bounds):
        """
        PLUCKER.intersect_volume Line intersection with volume
        
        P = self.intersect_volume(BOUNDS) is a matrix (3xN) with columns
        that indicate where the Plucker line PL intersects the faces of a volume
        specified by BOUNDS = [xmin xmax ymin ymax zmin zmax].  The number of
        columns N is either 0 (the line is outside the plot volume) or 2 (where
        the line pierces the bounding volume).
        
        [P,lambda] = self.intersect_volume(bounds, line) as above but also returns the
        line parameters (1xN) at the intersection points, ie. X = self.point(lambda).
        
        See also Plucker.plot, Plucker.point.
        """
        
        intersections = []
        
        # reshape, top row is minimum, bottom row is maximum
        bounds = bounds.reshape((2,3))
        
        for face in range(0, 6):
            # for each face of the bounding volume
            #  x=xmin, x=xmax, y=ymin, y=ymax, z=zmin, z=zmax
            
            i = math.ceil(face / 2)  # 1,2,3
            I = np.eye(3,3)
            p = [0, 0, 0]
            p[i] = bounds[face]
            plane = Plane(I[:,i], p)
            
            # find where line pierces the plane
            try:
                p,lam = line.intersect_plane(plane)
            except TypeError:
                continue  # no intersection with this plane
            
#            print('face %d: n=(%f, %f, %f), p=(%f, %f, %f)' % (face, plane.n, plane.p))
#            print('      : p=(%f, %f, %f)  ' % p)
            
            # find if intersection point is within the cube face
            #  test x,y,z simultaneously
            k = (p >= bounds[0,:]) & (p <= bounds[1,:])
            del k[i]  # remove the boolean corresponding to current face
            if all(k):
                # if within bounds, add
                intersections.append(lam)
                
#                     print('  HIT');

        # put them in ascending order
        intersections.sort()
        
        p = line.point(intersections)
        
        return namedtuple('intersect_volume', 'p lam')(p, intersections)

    
    # ------------------------------------------------------------------------- #
    #  PLOT AND DISPLAY
    # ------------------------------------------------------------------------- #   
    
    def plot(line, bounds=None, **kwargs):
        """
        Plucker.plot Plot a line

        self.plot(OPTIONS) adds the Plucker line PL to the current plot volume.

        self.plot(B, OPTIONS) as above but plots within the plot bounds B = [XMIN
        XMAX YMIN YMAX ZMIN ZMAX].

        Options::
         - Are passed directly to plot3, eg. 'k--', 'LineWidth', etc.

        Notes::
         - If the line does not intersect the current plot volume nothing will
           be displayed.

        See also plot3, Plucker.intersect_volume.
        """
        
        if bounds is None:
            ax = plt.gca()
            bounds = np.r_[ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]
        else:
            ax.set_xlim(bounds[:2])
            ax.set_ylim(bounds[2:4])
            ax.set_zlim(bounds[4:6])
        
        #U = self.Q - self.P;
        #line.p = self.P; line.v = unit(U);
        
        P, lam = line.intersect_volume(bounds)
        
        if len(lam) == 0:
            print('line does not intersect the plot volume')
        else:
            plt.plot(P[0,:], P[1,:], P[2,:], **kwargs)

    
    def __str__(self):
        """
        Plucker.char Convert to string

        s = P.char() is a string showing Plucker parameters in a compact single
        line format.

        See also Plucker.display.
        """
        
        return '{{ {:.5g} {:.5g} {:.5g}; {:.5g} {:.5g} {:.5g}}}'.format(*list(self.vec))

    def __repr__(self):
        return self.__str__()
        
        
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



if __name__ == "__main__":
    
    pl = Plucker.PQ([0, 0, 0], [1, 2, 3])
    print(pl)

    import unittest
    import numpy.testing as nt


    class PluckerTest(unittest.TestCase):
        
    
        # Primitives
        def test_constructor1(self):
            
            # construct from 6-vector
            L = Plucker([1, 2, 3, 4, 5, 6])
            self.assertIsInstance(L, Plucker)
            nt.assert_array_almost_equal(L.v, np.r_[1, 2, 3])
            nt.assert_array_almost_equal(L.w, np.r_[4, 5, 6])
            
            # construct from object
            L2 = Plucker(L)
            self.assertIsInstance(L, Plucker)
            nt.assert_array_almost_equal(L2.v, np.r_[1, 2, 3])
            nt.assert_array_almost_equal(L2.w, np.r_[4, 5, 6])
            
            # construct from point and direction
            L = Plucker.PointDir([1, 2, 3], [4, 5, 6])
            self.assertTrue(L.contains([1, 2, 3]))
            nt.assert_array_almost_equal(L.uw, sm.unitvec([4, 5, 6]))
        
        
        def test_vec(self):
            # verify double
            L = Plucker([1, 2, 3, 4, 5, 6])
            nt.assert_array_almost_equal(L.vec, np.r_[1, 2, 3, 4, 5, 6])
        
        def test_constructor2(self):
            # 2, point constructor
            P = np.r_[2, 3, 7]
            Q = np.r_[2, 1, 0]
            L = Plucker.PQ(P, Q)
            nt.assert_array_almost_equal(L.w, P-Q)
            nt.assert_array_almost_equal(L.v, np.cross(P-Q, Q))
        
            # TODO, all combos of list and ndarray
            # test all possible input shapes
            # L2, = Plucker(P, Q)
            # self.assertEqual(double(L2), double(L))
            # L2, = Plucker(P, Q')
            # self.assertEqual(double(L2), double(L))
            # L2, = Plucker(P', Q')
            # self.assertEqual(double(L2), double(L))
            # L2, = Plucker(P, Q)
            # self.assertEqual(double(L2), double(L))
            
            # # planes constructor
            # P = [10, 11, 12]'; w = [1, 2, 3]
            # L = Plucker.PointDir(P, w)
            # self.assertEqual(double(L), [cross(w,P) w]'); %FAIL
            # L2, = Plucker.PointDir(P', w)
            # self.assertEqual(double(L2), double(L))
            # L2, = Plucker.PointDir(P, w')
            # self.assertEqual(double(L2), double(L))
            # L2, = Plucker.PointDir(P', w')
            # self.assertEqual(double(L2), double(L))
        
        
        def test_pp(self):
            # validate pp and ppd
            L = Plucker.PQ([-1, 1, 2], [1, 1, 2])
            nt.assert_array_almost_equal(L.pp, np.r_[0, 1, 2])
            self.assertEqual(L.ppd, math.sqrt(5))
            
            # validate pp
            self.assertTrue( L.contains(L.pp) )
        
        
        def test_contains(self):
            P = [2, 3, 7]
            Q = [2, 1, 0]
            L = Plucker.PQ(P, Q)
            
            # validate contains
            self.assertTrue( L.contains([2, 3, 7]) )
            self.assertTrue( L.contains([2, 1, 0]) )
            self.assertFalse( L.contains([2, 1, 4]) )
        
        
        def test_closest(self):
            P = [2, 3, 7]
            Q = [2, 1, 0]
            L = Plucker.PQ(P, Q)
            
            out = L.closest(P)
            nt.assert_array_almost_equal(out.p, np.c_[P])
            self.assertEqual(out.d, 0)
            
             # validate closest with given points and origin
            out = L.closest(Q)
            nt.assert_array_almost_equal(out.p, Q)
            self.assertEqual(out.d, 0)
            
            L = Plucker.PQ([-1, 1, 2], [1, 1, 2])
            out = L.closest([0, 1, 2])
            nt.assert_array_almost_equal(out.p, np.r_[0, 1, 2])
            self.assertEqual(out.d, 0)
            
            out = L.closest([5, 1, 2])
            nt.assert_array_almost_equal(out.p, np.r_[5, 1, 2])
            self.assertEqual(out.d, 0)
            
            out = L.closest([0, 0, 0])
            snt.assert_array_almost_equal(out.p, L.pp)
            self.assertEqual(out.d, L.ppd)
            
            out = L.closest([5, 1, 0])
            snt.assert_array_almost_equal(out.pp, [5, 1, 2])
            self.assertEqual(out.d, 2)
        
        def test_plot(self):
            
            P = [2, 3, 7]
            Q = [2, 1, 0]
            L = Plucker.PQ(P, Q)
            
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
            ax.set_xlim3d(-10, 10)
            ax.set_ylim3d(-10, 10)
            ax.set_zlim3d(-10, 10)
            
            L.plot(linecolor='red', linewidth=2)
        
        def test_eq(self):
            w = np.r_[1, 2, 3]
            P = np.r_[-2, 4, 3]
            
            L1 = Plucker.PQ(P, P + w)
            L2 = Plucker.PQ(P + 2 * w, P + 5 * w)
            L3 = Plucker.PQ(P + np.r_[1, 0, 0], P + w)
            
            self.assertTrue(L1 == L2)
            self.assertFalse(L1 == L3)
            
            self.assertFalse(L1 != L2)
            self.assertTrue(L1 != L3)
        
        def test_skew(self):
            
            P = [2, 3, 7]; Q = [2, 1, 0]
            L = Plucker.PQ(P, Q)
            
            m = L.skew
            
            self.assertEqual(m.shape, (4,4))
            nt.assert_array_almost_equal(m + m.T, np.zeros((4,4)))
        
        def test_mtimes(self):
            P = [1, 2, 0]
            Q = [1, 2, 10]  # vertical line through (1,2)
            L = Plucker.PQ(P, Q)
            
            # check pre/post multiply by matrix
            M = np.random.uniform(size=(4,10))
            
            a = L * M
            nt.assert_array_almost_equal(a, L.skew @ M)
            
            M = np.random.uniform(size=(10,4))
            a = M * L
            nt.assert_array_almost_equal(a, M @ L.skew)
            
            # check transformation by SE3
            
            L2 = SE3() @ L
            nt.assert_array_almost_equal(L.vec, L2.vec)
            
            L2 = SE3(2, 3, 1) @ L # shift line in the xy directions
            pxy = L2.intersect_plane([0, 0, 1, 0])
            nt.assert_array_almost_equal(pxy, np.r_[1+2, 2+3, 0])
        
        def test_parallel(self):
            
            L1 = Plucker.PointDir([4, 5, 6], [1, 2, 3])
            L2 = Plucker.PointDir([5, 5, 6], [1, 2, 3])
            L3 = Plucker.PointDir([4, 5, 6], [3, 2, 1])
            
            # L1, || L2, but doesnt intersect
            # L1, intersects L3
            
            self.assertTrue( L1.isparallel(L1) )
            self.assertTrue(L1 | L1)
            
            self.assertTrue( L1.isparallel(L2) )
            self.assertTrue(L1 | L2)
            self.assertTrue( L2.isparallel(L1) )
            self.assertTrue(L2 | L1)
            self.assertFalse( L1.isparallel(L3) )
            self.assertFalse(L1 | L3)
        
        
        def test_intersect(self):
        
            
            L1 = Plucker.PointDir([4, 5, 6], [1, 2, 3])
            L2 = Plucker.PointDir([5, 5, 6], [1, 2, 3])
            L3 = Plucker.PointDir( [4, 5, 6], [0, 0, 1])
            L4 = Plucker.PointDir([5, 5, 6], [1, 0, 0])
        
            # L1, || L2, but doesnt intersect
            # L3, intersects L4
            self.assertFalse( L1^L2, )
            
            self.assertTrue( L3^L4, )
            
            
        def test_commonperp(self):
            L1 = Plucker.PointDir([4, 5, 6], [0, 0, 1])
            L2 = Plucker.PointDir([6, 5, 6], [0, 1, 0])
            
            self.assertFalse( L1|L2)
            self.assertFalse( L1^L2)
            
            self.assertEqual( distance(L1, L2), 2)
            
            L = L1.commonperp(L2)  # common perp intersects both lines
            
            self.assertTrue( L^L1)
            self.assertTrue( L^L2)
        
        
        def test_line(self):
            
            # mindist
            # intersect
            # char
            # intersect_volume
            # mindist
            # mtimes
            # or
            # side
            pass
        
        def test_point(self):
            P = [2, 3, 7]
            Q = [2, 1, 0]
            L = Plucker.PQ(P, Q)
            
            self.assertTrue( L.contains(L.point(0)) )
            self.assertTrue( L.contains(L.point(1)) )
            self.assertTrue( L.contains(L.point(-1)) )
        
        
        def test_char(self):
            P = [2, 3, 7]
            Q = [2, 1, 0]
            L = Plucker.PQ(P, Q)
            
            s = str(L)
            self.assertIsInstance(s, str)
    
        def test_plane(self):
            
            xyplane = [0, 0, 1, 0]
            xzplane = [0, 1, 0, 0]
            L = Plucker.Planes(xyplane, xzplane) # x axis
            nt.assert_array_almost_equal(L.vec, np.r_[0, 0, 0, -1, 0, 0])
            
            L = Plucker.PQ([-1, 2, 3], [1, 2, 3]);  # line at y=2,z=3
            x6 = [1, 0, 0, -6]  # x = 6
            
            # plane_intersect
            p,lam = L.intersect_plane(x6)
            nt.assert_array_almost_equal(p, np.r_[6, 2, 3])
            nt.assert_array_almost_equal(L.point(lam), np.r_[6, 2, 3])
            
    
            x6s = Plane(n=[1, 0, 0], p=[6, 0, 0])
            p,lam = L.intersect_plane(x6s)
            self.assertEqual(p, [6, 2, 3])
            self.assertEqual(L.point(lam), np.r_[6, 2, 3])
        
        def test_methods(self):
            # intersection
            px = Plucker.PQ(v=[0, 0, 0], w=[1, 0, 0]);  # x-axis
            py = Plucker.PQ(v=[0, 0, 0], w=[0, 1, 0]);  # y-axis
            px1 = Plucker.PQ(v=[0, 1, 0], w=[1, 1, 0]); # offset x-axis
            
            verifyEqual(tc, px.ppd, 0)
            verifyEqual(tc, px1.ppd, 1)
            verifyEqual(tc, px1.pp, [0, 1, 0])

            px.intersects(px)
            px.intersects(py)
            px.intersects(px1)
            
            
        # def test_intersect(self):
        #     px = Plucker([0, 0, 0], [1, 0, 0]);  # x-axis
        #     py = Plucker([0, 0, 0], [0, 1, 0]);  # y-axis
        #     
        #     plane.d = [1, 0, 0]; plane.p = 2; # plane x=2
        #     
        #     px.intersect_plane(plane)
        #     py.intersect_plane(plane)

    unittest.main()