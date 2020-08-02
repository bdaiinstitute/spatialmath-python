import numpy as np
import math

from spatialmath.base import argcheck
import spatialmath.base as tr
from spatialmath import super_pose as sp
from spatialmath import SO2, SE2


class Twist2(sp.SMTwist):
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

    # -------------------- variant constructors ----------------------------#

    @classmethod
    def R(cls, q):
        """
        Construct a new 2D revolute Twist object
        
        :param a: displacment
        :type a: 2-element array-like
        :return: 2D prismatic twist
        :rtype: Twist2 instance
        
        - ``Twist.R(q)`` is a 2D Twist object representing rotation about the 2D point ``q``.
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
        
        - ``Twist.P(q)`` is a 2D Twist object representing 2D-translation in the direction ``a``.
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

    exec(open(os.path.join(pathlib.Path(__file__).parent.absolute(), "test_twist2d.py")).read())
    