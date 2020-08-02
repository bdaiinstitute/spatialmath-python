import numpy.testing as nt
import matplotlib.pyplot as plt
import unittest

"""
we will assume that the primitives rotx,trotx, etc. all work
"""
from math import pi
from spatialmath.Twist import *
from spatialmath import super_pose as sp
from spatialmath.base import *
from spatialmath.base import argcheck


def array_compare(x, y):
    if isinstance(x, sp.SMPose):
        x = x.A
    if isinstance(y, sp.SMPose):
        y = y.A
    if isinstance(x, sp.SMTwist):
        x = x.S
    if isinstance(y, sp.SMTwist):
        y = y.S
    nt.assert_array_almost_equal(x, y)


class Twist3dTest(unittest.TestCase):
    
    def test_constructor(self):
        
        s = [1, 2, 3, 4, 5, 6]
        x = Twist(s)
        self.assertIsInstance(x, Twist)
        self.assertEqual(len(x), 1)
        array_compare(x.v, [1, 2, 3])
        array_compare(x.w, [4, 5, 6])
        array_compare(x.S, s)
        
        x = Twist([1,2,3], [4,5,6])
        array_compare(x.v, [1, 2, 3])
        array_compare(x.w, [4, 5, 6])
        array_compare(x.S, s)

        y = Twist(x)
        array_compare(x, y)
        
        x = Twist(SE3())
        array_compare(x, [0,0,0,0,0,0])
        
        
    def test_list(self):
        x = Twist([1, 0, 0, 0, 0, 0])
        y = Twist([1, 0, 0, 0, 0, 0])
                   
        a = Twist(x)
        a.append(y)
        self.assertEqual(len(a), 2)
        array_compare(a[0], x)
        array_compare(a[1], y)
        
    def test_conversion_SE3(self):
        T = SE3.Rx(0)
        tw = Twist(T)
        array_compare(tw.SE3(), T)
        self.assertIsInstance(tw.SE3(), SE3)
        self.assertEqual(len(tw.SE3()), 1)

        T = SE3.Rx(0) * SE3(1, 2, 3)
        array_compare(Twist(T).SE3(), T)
        
    def test_conversion_se3(self):
        s = [1, 2, 3, 4, 5, 6]
        x = Twist(s)
        
        array_compare(x.se3(), np.array([[ 0., -6.,  5.,  1.],
                    [ 6.,  0., -4.,  2.],
                    [-5.,  4.,  0.,  3.],
                    [ 0.,  0.,  0.,  0.]]))
        
    def test_conversion_Plucker(self):
        pass
        
    def test_list_constuctor(self):
        x = Twist([1, 0, 0, 0, 0, 0])
        
        a = Twist([x,x,x,x])
        self.assertIsInstance(a, Twist)
        self.assertEqual(len(a), 4)
        
        a = Twist([x.se3(), x.se3(), x.se3(), x.se3()])
        self.assertIsInstance(a, Twist)
        self.assertEqual(len(a), 4)
        
        a = Twist([x.S, x.S, x.S, x.S])
        self.assertIsInstance(a, Twist)
        self.assertEqual(len(a), 4)
        
        s = [1, 2, 3, 4, 5, 6]
        a = Twist([s, s, s, s])
        self.assertIsInstance(a, Twist)
        self.assertEqual(len(a), 4)
        
    def test_predicate(self):
        x = Twist.R([1, 2, 3], [0, 0, 0])
        self.assertFalse(x.isprismatic)
        
        # check prismatic twist
        x = Twist.P([1, 2, 3])
        self.assertTrue(x.isprismatic)
        
        self.assertTrue(Twist.isvalid(x.se3()))
        self.assertTrue(Twist.isvalid(x.S))
        
        self.assertFalse(Twist.isvalid(2))
        self.assertFalse(Twist.isvalid(np.eye(4)))
        
    def test_str(self):
        x = Twist([1, 2, 3, 4, 5, 6])
        s = str(x)
        self.assertIsInstance(s, str)
        self.assertEqual(len(s), 14)
        self.assertEqual(s.count('\n'), 0)
        
        x.append(x)
        s = str(x)
        self.assertIsInstance(s, str)
        self.assertEqual(len(s), 29)
        self.assertEqual(s.count('\n'), 1)
        
    def test_variant_constructors(self):
        
        # check rotational twist
        x = Twist.R([1, 2, 3], [0, 0, 0])
        array_compare(x, np.r_[0, 0, 0, unitvec([1, 2, 3])])
        
        # check prismatic twist
        x = Twist.P([1, 2, 3])
        array_compare(x, np.r_[unitvec([1, 2, 3]), 0, 0, 0, ])
    
    def test_SE3_twists(self):
        tw = Twist( SE3.Rx(0) )
        array_compare(tw, np.r_[0, 0, 0,  0, 0, 0])
                      
        tw = Twist( SE3.Rx(pi / 2) )
        array_compare(tw, np.r_[0, 0, 0,  pi / 2, 0, 0])
                      
        tw = Twist( SE3.Ry(pi / 2) )
        array_compare(tw, np.r_[0, 0, 0,  0, pi / 2, 0])
                      
        tw = Twist( SE3.Rz(pi / 2) )
        array_compare(tw, np.r_[0, 0, 0,  0, 0, pi / 2])
        
        tw = Twist( SE3([1, 2, 3]) )
        array_compare(tw, [1, 2, 3,  0, 0, 0])
                      
        tw = Twist( SE3([1, 2, 3]) * SE3.Ry(pi / 2))
        array_compare(tw, np.r_[-pi / 2, 2, pi,  0, pi / 2, 0])
        
    def test_exp(self):
        tw = Twist.R([1, 0, 0], [0, 0, 0])
        array_compare(tw.exp(pi/2), SE3.Rx(pi/2))
        
        tw = Twist.R([0, 1, 0], [0, 0, 0])
        array_compare(tw.exp(pi/2), SE3.Ry(pi/2))
        
        tw = Twist.R([0, 0, 1], [0, 0, 0])
        array_compare(tw.exp(pi/2), SE3.Rz(pi / 2))
    
    def test_arith(self):
        
        # check overloaded *

        T1 = SE3(1, 2, 3) * SE3.Rx(pi / 2)
        T2 = SE3(4, 5, -6) * SE3.Ry(-pi / 2)
        
        x1 = Twist(T1)
        x2 = Twist(T2)

        array_compare( (x1 * x2).exp(), T1 * T2)
        array_compare( (x2 * x1).exp(), T2 * T1)
        
    def test_prod(self):
        # check prod
        T1 = SE3(1, 2, 3) * SE3.Rx(pi / 2)
        T2 = SE3(4, 5, -6) * SE3.Ry(-pi / 2)
        
        x1 = Twist(T1)
        x2 = Twist(T2)
        
        x = Twist([x1, x2])
        array_compare( x.prod().SE3(), T1 * T2)
        



# ---------------------------------------------------------------------------------------#
if __name__ == '__main__':

    unittest.main()
