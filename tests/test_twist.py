import numpy.testing as nt
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt

"""
we will assume that the primitives rotx,trotx, etc. all work
"""
from math import pi
from spatialmath.twist import *
# from spatialmath import super_pose # as sp
from spatialmath.base import *
from spatialmath.baseposematrix import BasePoseMatrix
from spatialmath import SE2, SE3
from spatialmath.twist import BaseTwist

def array_compare(x, y):
    if isinstance(x, BasePoseMatrix):
        x = x.A
    if isinstance(y, BasePoseMatrix):
        y = y.A
    if isinstance(x, BaseTwist):
        x = x.S
    if isinstance(y, BaseTwist):
        y = y.S
    nt.assert_array_almost_equal(x, y)


class TestTwist3d:
    
    def test_constructor(self):
        
        s = [1, 2, 3, 4, 5, 6]
        x = Twist3(s)
        assert isinstance(x, Twist3)
        assert len(x) == 1
        array_compare(x.v, [1, 2, 3])
        array_compare(x.w, [4, 5, 6])
        array_compare(x.S, s)
        
        x = Twist3([1,2,3], [4,5,6])
        array_compare(x.v, [1, 2, 3])
        array_compare(x.w, [4, 5, 6])
        array_compare(x.S, s)

        y = Twist3(x)
        array_compare(x, y)
        
        x = Twist3(SE3())
        array_compare(x, [0,0,0,0,0,0])
        
    def test_conversion_SE3(self):
        T = SE3.Rx(0)
        tw = Twist3(T)
        array_compare(tw.SE3(), T)
        assert isinstance(tw.SE3(), SE3)
        assert len(tw.SE3()) == 1

        T = SE3.Rx(0) * SE3(1, 2, 3)
        array_compare(Twist3(T).SE3(), T)
        
    def test_conversion_se3(self):
        s = [1, 2, 3, 4, 5, 6]
        x = Twist3(s)
        
        array_compare(x.skewa(), np.array([[ 0., -6.,  5.,  1.],
                    [ 6.,  0., -4.,  2.],
                    [-5.,  4.,  0.,  3.],
                    [ 0.,  0.,  0.,  0.]]))
        
    def test_conversion_Plucker(self):
        pass
        
    def test_predicate(self):
        x = Twist3.UnitRevolute([1, 2, 3], [0, 0, 0])
        assert not x.isprismatic
        
        # check prismatic twist
        x = Twist3.UnitPrismatic([1, 2, 3])
        assert x.isprismatic
        
        assert Twist3.isvalid(x.skewa())
        assert Twist3.isvalid(x.S)
        
        assert not Twist3.isvalid(2)
        assert not Twist3.isvalid(np.eye(4))
        
    def test_str(self):
        x = Twist3([1, 2, 3, 4, 5, 6])
        s = str(x)
        assert isinstance(s, str)
        assert len(s) == 14
        assert s.count('\n') == 0
        
    def test_variant_constructors(self):
        
        # check rotational twist
        x = Twist3.UnitRevolute([1, 2, 3], [0, 0, 0])
        array_compare(x, np.r_[0, 0, 0, unitvec([1, 2, 3])])
        
        # check prismatic twist
        x = Twist3.UnitPrismatic([1, 2, 3])
        array_compare(x, np.r_[unitvec([1, 2, 3]), 0, 0, 0, ])
    
    def test_SE3_twists(self):
        tw = Twist3( SE3.Rx(0) )
        array_compare(tw, np.r_[0, 0, 0,  0, 0, 0])
                      
        tw = Twist3( SE3.Rx(pi / 2) )
        array_compare(tw, np.r_[0, 0, 0,  pi / 2, 0, 0])
                      
        tw = Twist3( SE3.Ry(pi / 2) )
        array_compare(tw, np.r_[0, 0, 0,  0, pi / 2, 0])
                      
        tw = Twist3( SE3.Rz(pi / 2) )
        array_compare(tw, np.r_[0, 0, 0,  0, 0, pi / 2])
        
        tw = Twist3( SE3([1, 2, 3]) )
        array_compare(tw, [1, 2, 3,  0, 0, 0])
                      
        tw = Twist3( SE3([1, 2, 3]) * SE3.Ry(pi / 2))
        array_compare(tw, np.r_[-pi / 2, 2, pi,  0, pi / 2, 0])
        
    def test_exp(self):
        tw = Twist3.UnitRevolute([1, 0, 0], [0, 0, 0])
        array_compare(tw.exp(pi/2), SE3.Rx(pi/2))
        
        tw = Twist3.UnitRevolute([0, 1, 0], [0, 0, 0])
        array_compare(tw.exp(pi/2), SE3.Ry(pi/2))
        
        tw = Twist3.UnitRevolute([0, 0, 1], [0, 0, 0])
        array_compare(tw.exp(pi/2), SE3.Rz(pi / 2))
    
    def test_arith(self):
        
        # check overloaded *

        T1 = SE3(1, 2, 3) * SE3.Rx(pi / 2)
        T2 = SE3(4, 5, -6) * SE3.Ry(-pi / 2)
        
        x1 = Twist3(T1)
        x2 = Twist3(T2)

        array_compare( (x1 * x2).exp(), T1 * T2)
        array_compare( (x2 * x1).exp(), T2 * T1)
        
    def test_prod(self):
        # check prod
        T1 = SE3(1, 2, 3) * SE3.Rx(pi / 2)
        T2 = SE3(4, 5, -6) * SE3.Ry(-pi / 2)
        
        x1 = Twist3(T1)
        x2 = Twist3(T2)
        
        x = Twist3([x1, x2])
        array_compare( x.prod().SE3(), T1 * T2)
        

class TestTwist2d:
    
    def test_constructor(self):
        
        s = [1, 2, 3]
        x = Twist2(s)
        assert isinstance(x, Twist2)
        assert len(x) == 1
        array_compare(x.v, [1, 2])
        array_compare(x.w, [3])
        array_compare(x.S, s)
        
        x = Twist2([1,2], 3)
        array_compare(x.v, [1, 2])
        array_compare(x.w, [3])
        array_compare(x.S, s)

        y = Twist2(x)
        array_compare(x, y)
        
        # construct from SE2
        x = Twist2(SE2())
        array_compare(x, [0,0,0])
        
        x = Twist2( SE2(0, 0, pi / 2))
        array_compare(x, np.r_[0, 0, pi / 2])
        
        x = Twist2( SE2(1, 2,0 ))
        array_compare(x, np.r_[1, 2, 0])
        
        x = Twist2( SE2(1, 2, pi / 2))
        array_compare(x, np.r_[3 * pi / 4, pi / 4, pi / 2])
        
    def test_variant_constructors(self):
        
        # check rotational twist
        x = Twist2.UnitRevolute([1, 2])
        array_compare(x, np.r_[2, -1, 1])
        
        # check prismatic twist
        x = Twist2.UnitPrismatic([1, 2])
        array_compare(x, np.r_[unitvec([1, 2]), 0])
        
    def test_conversion_SE2(self):
        T = SE2(1, 2, 0.3)
        tw = Twist2(T)
        array_compare(tw.SE2(), T)
        assert isinstance(tw.SE2(), SE2)
        assert len(tw.SE2()) == 1
        
    def test_conversion_se2(self):
        s = [1, 2, 3]
        x = Twist2(s)
        
        array_compare(x.skewa(), np.array([[ 0., -3.,  1.],
                                       [ 3.,  0.,  2.],
                                       [ 0.,  0.,  0.]]))

    def test_predicate(self):
        x = Twist2.UnitRevolute([1, 2])
        assert not x.isprismatic
        
        # check prismatic twist
        x = Twist2.UnitPrismatic([1, 2])
        assert x.isprismatic
        
        assert Twist2.isvalid(x.skewa())
        assert Twist2.isvalid(x.S)
        
        assert not Twist2.isvalid(2)
        assert not Twist2.isvalid(np.eye(3))
        
    def test_str(self):
        x = Twist2([1, 2, 3])
        s = str(x)
        assert isinstance(s, str)
        assert len(s) == 8
        assert s.count('\n') == 0

    def test_SE2_twists(self):
        tw = Twist2( SE2() )
        array_compare(tw, np.r_[0, 0, 0])
                      
        tw = Twist2( SE2(0, 0, pi / 2) )
        array_compare(tw, np.r_[0, 0, pi / 2])
                      
        
        tw = Twist2( SE2([1, 2, 0]) )
        array_compare(tw, [1, 2, 0])
                      
        tw = Twist2( SE2([1, 2, pi / 2]))
        array_compare(tw, np.r_[ 3 * pi / 4, pi / 4, pi / 2])
        
    def test_exp(self):
        x = Twist2.UnitRevolute([0, 0])
        array_compare(x.exp(pi/2), SE2(0, 0, pi/2))
        
        x = Twist2.UnitRevolute([1, 0])
        array_compare(x.exp(pi/2), SE2(1, -1, pi/2))
        
        x = Twist2.UnitRevolute([1, 2])
        array_compare(x.exp(pi/2), SE2(3, 1, pi/2))

    
    def test_arith(self):
        
        # check overloaded *

        T1 = SE2(1, 2, pi / 2)
        T2 = SE2(4, 5, -pi / 4)
        
        x1 = Twist2(T1)
        x2 = Twist2(T2)

        array_compare( (x1 * x2).exp(), (T1 * T2).A)
        array_compare( (x2 * x1).exp(), (T2 * T1).A)

        array_compare( (x1 * x2).SE2(), (T1 * T2).A)
        array_compare( (x2 * x1).SE2(), (T2 * T1))
        
    def test_prod(self):
        # check prod
        T1 = SE2(1, 2, pi / 2)
        T2 = SE2(4, 5, -pi / 4)
        
        x1 = Twist2(T1)
        x2 = Twist2(T2)
        
        x = Twist2([x1, x2])
        array_compare( x.prod().SE2(), T1 * T2)
