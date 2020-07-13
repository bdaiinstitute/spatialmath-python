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
import spatialmath.pose3d as p3

# ============================== SO2 =====================================#

class SO2(sp.SMPose):

    # SO2()  identity matrix
    # SO2(angle, unit)
    # SO2( obj )   # deep copy
    # SO2( np )  # make numpy object
    # SO2( nplist )  # make from list of numpy objects

    # constructor needs to take ndarray -> SO2, or list of ndarray -> SO2
    def __init__(self, arg=None, *, unit='rad', check=True):
        """
        Construct new SO(2) object

        :param arg: DESCRIPTION, defaults to None
        :type arg: TYPE, optional
        :param *: DESCRIPTION
        :type *: TYPE
        :param unit: DESCRIPTION, defaults to 'rad'
        :type unit: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE

        - ``SO2()`` is am SO2 instance representing a null rotation -- the identity matrix
        - ``SO2(theta)`` is an SO3 instance representing a rotation by ``theta``.  If ``theta`` is array_like
          `[theta1, theta2, ... thetaN]` then an SO2 instance containing N rotations.
        - ``SO2(R)`` is an SO3 instance with rotation matrix R which is a 2x2 numpy array representing an valid rotation matrix.  If ``check``
          is ``True`` check the matrix value.
        - ``SO2([R1, R2, ... RN])`` where each Ri is a 2x2 numpy array of rotation matrices, is
          an SO2 instance containing N rotations. If ``check`` is ``True``
          then each matrix is checked for validity.
        - ``SO2([R1, R2, ... RN])`` where each Ri is an SO2 instance, is an SO2 instance containing N rotations.
        """
        super().__init__()  # activate the UserList semantics

        if arg is None:
            # empty constructor
            if type(self) is SO2:
                self.data = [np.eye(2)]
        elif argcheck.isvector(arg):
            # SO2(value)
            # SO2(list of values)
            self.data = [tr.rot2(x, unit) for x in argcheck.getvector(arg)]

        elif isinstance(arg, np.ndarray) and arg.shape == (2,2):
            self.data = [arg]
        else:
            super().pose_arghandler(arg, check=check)

    @classmethod
    def Rand(cls, *, range=[0, 2 * math.pi], unit='rad', N=1):
        """
        Create SO(2) with random rotation

        :param N: number of random rotations
        :type N: int
        :return: SO(2) rotation matrix
        :rtype: SO2 instance

        - ``SO2.Rand()`` is a random SO(2) rotation.
        - ``SO2.Rand(N)`` is a sequence of N random rotations.

        """
        rand = np.random.uniform(low=range[0], high=range[1], size=N)  # random values in the range
        return cls([tr.rot2(x) for x in argcheck.getunit(rand, unit)])
    
    @classmethod
    def Exp(cls, S, so2=False):
        """
        Create an SO(2) rotation matrix from so(2)

        :param S: Lie algebra so(2)
        :type S: numpy ndarray
        :param so3: accept input as an so(2) matrix [default False]
        :type so3: bool
        :return: SO(2) rotation matrix
        :rtype: SO2 instance

        - ``SO2.Exp(S)`` is an SO(2) rotation defined by its Lie algebra
          which is a 2x2 so(2) matrix (skew symmetric)
        - ``SO2.Exp(t)`` is an SO(2) rotation defined by a 3-element twist
          vector (the unique elements of the so(2) skew-symmetric matrix)
        - ``SO2.Exp(T)`` is a sequence of SO(3) rotations defined by an Nx3 matrix
          of twist vectors, one per row.
          
        Note:
            
        - an input 3x3 matrix is ambiguous, it could be the first or third case above.  In this
          case the parameter `so3` is the decider.

        :seealso: :func:`spatialmath.base.transforms3d.trexp`, :func:`spatialmath.base.transformsNd.skew`
        """
        if argcheck.ismatrix(S, (-1,2)) and not so2:
            return cls([tr.trexp2(s) for s in S])
        else:
            return cls(tr.trexp2(S), check=False)

    @staticmethod
    def isvalid(x):
        """
        Test if matrix is valid SO(2)

        :param x: matrix to test
        :type x: numpy.ndarray
        :return: true of the matrix is 3x3 and a valid element of SO(2), ie. it is an
            orthonormal matrix with determinant of +1.
        :rtype: bool

        :seealso: :func:`~spatialmath.base.transform3d.isrot`
        """
        return tr.isrot2(x, check=True)

    @property
    def inv(self):
        """
        Inverse of SO(2)

        :param self: pose
        :type self: SE2 instance
        :return: inverse
        :rtype: SO2

        Returns the inverse, which for elements of SO(2) is the transpose.
        """
        if len(self) == 1:
            return SO2(self.A.T)
        else:
            return SO2([x.T for x in self.A])

    @property
    def R(self):
        """
        Rotational component as a matrix

        :param self: SO(2), SE(2)
        :type self: SO2 or SE2 instance
        :return: rotational component
        :rtype: numpy.ndarray

        ``T.R`` returns an:

        - ndarray with shape=(2,2), if len(T) == 1
        - ndarray with shape=(N,2,2), if len(T) = N > 1
        """
        return self.A[:2, :2]

    @property
    def theta(self):
        """Returns angle of SO2 object matrices in unit radians"""
        if len(self) == 1:
            return math.atan2(self.A[1,0], self.A[0,0])
        else:
            return [math.atan2(x.A[1,0], x.A[0,0]) for x in self]
    
    @property
    def SE2(self):
        """
        
        :return: SE(2) with same rotation but zero translation
        :rtype: SE2

        """
        return SE2(tr.rt2tr(self.A, [0, 0]))
    
    @property
    def log(self):
        return tr.trlog2(self.A)
    

# ============================== SE2 =====================================#

class SE2(SO2):
    # constructor needs to take ndarray -> SO2, or list of ndarray -> SO2
    def __init__(self, x=None, y=None, theta=None, *, unit='rad', check=True):
        """
        Construct new SE(2) object

        :param x: translation distance along the X-axis
        :type x: float
        :param y: translation distance along the Y-axis
        :type y: float
        :return: homogeneous transformation matrix
        :rtype: SE2 instance
        
        - ``SE2()`` is a null motion -- the identity matrix
        - ``SE2(x, y)`` is a pure translation of (x,y)
        - ``SE2(t)`` where ``t=[x,y]`` is a 2-element array_like, is a pure translation of (x,y)
        - ``SE2(x, y, theta)`` is a translation of (x,y) and a rotation of theta
        - ``SE2(t)`` where ``t=[x,y,theta]`` is a 3-element array_like, is a a translation of (x,y) and a rotation of theta
        - ``SE3(T)`` where T is a 3x3 numpy array representing an SE(2) matrix.  If ``check``
          is ``True`` check the matrix belongs to SE(2).
        - ``SE2([T1, T2, ... TN])`` where each Ti is a 3x3 numpy array representing an SE(2) matrix, is
          an SE2 instance containing N rotations. If ``check`` is ``True``
          check the matrix belongs to SE(2).
        - ``SE2([T1, T2, ... TN])`` where each Ri is an SE2 instance, is an SE2 instance containing N rotations.
        """
        super().__init__()  # activate the UserList semantics

        if x is None and y is None and theta is None:
            # SE2()
            # empty constructor
            self.data = [np.eye(3)]

        elif x is not None:
            if y is not None and theta is None:
                # SE2(x, y)
                self.data = [tr.transl2(x, y)]
            elif y is not None and theta is not None:
                # SE2(x, y, theta)
                self.data = [tr.trot2(theta, t=[x, y], unit=unit)]
            elif y is None and theta is None:
                if argcheck.isvector(x, 2):
                    # SE2([x,y])
                    self.data = [tr.transl2(x)]
                elif argcheck.isvector(x, 3):
                    # SE2([x,y,theta])
                    self.data = [tr.trot2(x[2], t=x[:2], unit=unit)]
                else:
                    super().pose_arghandler(x, check=check)
        else:
            raise ValueError('bad arguments to constructor')

    @property
    def T(self):
        raise NotImplemented('transpose is not meaningful for SE3 object')

    @classmethod
    def Rand(cls, *, xrange=[-1, 1], yrange=[-1, 1], trange=[0, 2 * math.pi], unit='rad', N=1):
        """
        Create a random SE(2)
    
        :param xrange: x-axis range [min,max], defaults to [-1, 1]
        :type xrange: 2-element sequence, optional
        :param yrange: y-axis range [min,max], defaults to [-1, 1]
        :type yrange: 2-element sequence, optional
        :param N: number of random rotations
        :type N: int
        :return: homogeneous transformation matrix
        :rtype: SE2 instance
    
        Return an SE2 instance with random rotation and translation.

        - ``SE2.Rand()`` is a random SE(2) rotation.
        - ``SE2.Rand(N)`` is an SE2 object containing a sequence of N random
          poses.
    
        """
        x = np.random.uniform(low=xrange[0], high=xrange[1], size=N)  # random values in the range
        y = np.random.uniform(low=yrange[0], high=yrange[1], size=N)  # random values in the range
        theta = np.random.uniform(low=trange[0], high=trange[1], size=N)  # random values in the range
        return cls([tr.trot2(t, t=[x, y]) for (t, x, y) in zip(x, y, argcheck.getunit(theta, unit))])

    @classmethod
    def Exp(cls, S):
        """
        Create an SE(2) rotation matrix from se(2)

        :param S: Lie algebra se(2)
        :type S: numpy ndarray
        :return: homogeneous transform matrix
        :rtype: SE2 instance

        - ``SE2.Exp(S)`` is an SE(2) rotation defined by its Lie algebra
          which is a 3x3 se(2) matrix (skew symmetric)
        - ``SE2.Exp(t)`` is an SE(2) rotation defined by a 3-element twist
          vector (the unique elements of the se(2) skew-symmetric matrix)

        :seealso: :func:`spatialmath.base.transforms3d.trexp`, :func:`spatialmath.base.transformsNd.skew`
        """
        # if isinstance(S, np.ndarray) and S.shape[1] == 3:
        #     return cls([tr.trexp2(s) for s in S])
        # else:
        # code above is problematic!
        #  cant tell an Nx3 from a 3x3 matrix, need an istwist
        return cls(tr.trexp2(S), check=False)
        
    @staticmethod
    def isvalid(x):
        """
        Test if matrix is valid SE(2)

        :param x: matrix to test
        :type x: numpy.ndarray
        :return: true of the matrix is 4x4 and a valid element of SE(2), ie. it is an
            homogeneous transformation matrix.
        :rtype: bool

        :seealso: :func:`~spatialmath.base.transform2d.ishom`
        """
        return tr.ishom2(x, check=True)

    @property
    def t(self):
        """
        Translational component of SE(2)

        :param self: SE(2)
        :type self: SE2 instance
        :return: translational component
        :rtype: numpy.ndarray

        ``T.t`` returns an:

        - ndarray with shape=(2,), if len(T) == 1
        - ndarray with shape=(N,2), if len(T) = N > 1
        """
        if len(self) == 1:
            return self.A[:2, 2]
        else:
            return np.array([x[:2, 2] for x in self.A])
    
    
    @property
    def xyt(self):
        """
        Configuration vector
        
        :return: An array :math:`[x, y, \theta]`
        :rtype: numpy.ndarray
        
        ``T.xyt`` returns an:
            
        - ndarray with shape=(3,), if len(T) == 1
        - ndarray with shape=(N,3), if len(T) = N > 1
        """
        if len(self) == 1:
            return np.r_[self.t, self.theta]
        else:
            return [np.r_[x.t, x.theta] for x in self]

    @property
    def inv(self):
        r"""
        Inverse of SE(2)

        :param self: pose
        :type self: SE2 instance
        :return: inverse
        :rtype: SE2

        Returns the inverse taking into account its structure

        :math:`T = \left[ \begin{array}{cc} R & t \\ 0 & 1 \end{array} \right], T^{-1} = \left[ \begin{array}{cc} R^T & -R^T t \\ 0 & 1 \end{array} \right]`
        """
        if len(self) == 1:
            return SE2(tr.rt2tr(self.R.T, -self.R.T @ self.t))
        else:
            return SE2([tr.rt2tr(x.R.T, -x.R.T @ x.t) for x in self])
        
    @property
    def SE3(self):
        def lift3(x):
            y = np.eye(4)
            y[:2,:2] = x.A[:2,:2]
            y[:2,3] = x.A[:2,2]
            return y
        return p3.SE3([lift3(x) for x in self])
    
    @property
    def Twist(self):
        return Twist2

# ============================== Twist =====================================#

class Twist2(sp.SMTwist):
    def __init__(self, arg=None, w=None, check=True):
        """
        Create 2D Twist object

        TW = Twist2(T) is a Twist object representing the SE(2) or SE(3)
        homogeneous transformation matrix T (3x3 or 4x4).

        TW = Twist2(V) is a twist object where the vector is specified directly.

        3D CASE:

        TW = Twist('R', A, Q) is a Twist object representing rotation about the
        axis of direction A (3x1) and passing through the point Q (3x1).
                %
        TW = Twist('R', A, Q, P) as above but with a pitch of P (distance/angle).

        TW = Twist('T', A) is a Twist object representing translation in the
        direction of A (3x1).

        2D CASE:

        TW = Twist('R', Q) is a Twist object representing rotation about the point Q (2x1).

        TW = Twist('T', A) is a Twist object representing translation in the
        direction of A (2x1).

        Notes:

        - The argument 'P' for prismatic is synonymous with 'T'.
        """

        super().__init__()   # enable UserList superpowers

        if arg is None:
            self.data = [np.r_[0.0, 0.0, 0.0,]]
        
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

    @classmethod
    def R(cls, q):
        
        q = argcheck.getvector(q, 2)
        v = -np.cross(np.r_[0.0, 0.0, 1.0], np.r_[q, 0.0])

        return cls(v[:2], 1)

    @classmethod
    def P(cls, a):
        w = 0
        v = tr.unitvec(argcheck.getvector(a, 2))

        return cls(v, w)
    
    @property
    def v(self):
        return self.data[0][:2]
    
    @property
    def w(self):
        return self.data[0][2]
    
    # ------------------------- static methods -------------------------------#

    @staticmethod
    def isvalid(v, check=True):
        if argcheck.isvector(v, 3):
            return True
        elif argcheck.ismatrix(v, (3,3)):
            # maybe be an se(2)
            if not all(v.diagonal() == 0):  # check diagonal is zero 
                return False
            if not all(v[2,:] == 0):  # check bottom row is zero
                return False
            if not tr.isskew(v[:2,:2]):
                  # top left 2x2is skew symmetric
                  return False
            return True
        return False

    @property
    def SE2(tw):
        """
        %Twist.SE Convert twist to SE2 or SE3 object
        %
        TW.SE is an SE2 or SE3 object representing the homogeneous transformation equivalent to the twist.
                %
            See also Twist.T, SE2, SE3.
        """

        return SE2( tw.exp() )
    
    @property
    def se2(self):
        """
        Twist.se Return the twist matrix

        TW.se is the twist matrix in se(2) or se(3) which is an augmented
        skew-symmetric matrix (3x3 or 4x4).

        """
        if len(self) == 1:
            return tr.skewa(self.S)
        else:
            return [tr.skewa(x.S) for x in self]
        
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
            return SE2(tr.trexp2(self.S *  theta))
        else:
            return SE2([tr.trexp2(self.S *  t) for t in theta])
        
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
        Twist.ad Logarithm of adjoint

        TW.ad is the logarithm of the adjoint matrix of the corresponding
        homogeneous transformation.

        See also SE3.Ad.
        """
        x = np.array([skew(self.w), skew(self.v), [np.zeros((3,3)), skew(self.w)]])
        
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
        
        if isinstance(right, Twist2):
            # twist composition
            return Twist2( left.exp() * right.exp());
        elif isinstance(right, (int, np.int64, float, np.float64)):
            return Twist2(left.S * right)
        else:
            raise ValueError('twist *, incorrect right operand')

    def __imul__(left,right):
        return left.__mul__(right)

    def __rmul(right, left):
        if isinstance(left, (int, np.int64, float, np.float64)):
            return Twist2(right.S * left)
        else:
            raise ValueError('twist *, incorrect left operand')
            
    def __str__(self):
        """
    %Twist.char Convert to string

    s = TW.char() is a string showing Twist parameters in a compact single line format.
    If TW is a vector of Twist objects return a string with one line per Twist.

    See also Twist.display.
        """
        return '\n'.join(["({:.5g} {:.5g}; {:.5g})".format(*list(tw.S)) for tw in self])

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
            return "Twist2([{:.5g}, {:.5g}, {:.5g}])".format(*list(self.S))
        else:
            return "Twist2([\n" + \
                ',\n'.join(["  [{:.5g}, {:.5g}, {:.5g}}]".format(*list(tw.S)) for tw in self]) +\
                "\n])"


if __name__ == '__main__':  # pragma: no cover

    import pathlib
    import os.path

    #exec(open(os.path.join(pathlib.Path(__file__).parent.absolute(), "test_pose2d.py")).read())
    
    a = SE2()
    a.interp(a, 0)

    