import numpy.testing as nt
import matplotlib.pyplot as plt
import unittest

"""
we will assume that the primitives rotx,trotx, etc. all work
"""
from math import pi
from spatialmath.Twist2 import *
from spatialmath.pose2d import *
from spatialmath import super_pose as sp
from spatialmath.base import *
import spatialmath.base.argcheck as argcheck


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



class Twist2dTest(unittest.TestCase):
    
    def test_constructor(self):
        
        s = [1, 2, 3]
        x = Twist2(s)
        self.assertIsInstance(x, Twist2)
        self.assertEqual(len(x), 1)
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
        
        
    def test_list(self):
        x = Twist2([1, 0, 0])
        y = Twist2([1, 0, 0])
                   
        a = Twist2(x)
        a.append(y)
        self.assertEqual(len(a), 2)
        array_compare(a[0], x)
        array_compare(a[1], y)

    def test_variant_constructors(self):
        
        # check rotational twist
        x = Twist2.R([1, 2])
        array_compare(x, np.r_[2, -1, 1])
        
        # check prismatic twist
        x = Twist2.P([1, 2])
        array_compare(x, np.r_[unitvec([1, 2]), 0])
        
    def test_conversion_SE2(self):
        T = SE2(1, 2, 0.3)
        tw = Twist2(T)
        array_compare(tw.SE2, T)
        self.assertIsInstance(tw.SE2, SE2)
        self.assertEqual(len(tw.SE2), 1)
        
    def test_conversion_se2(self):
        s = [1, 2, 3]
        x = Twist2(s)
        
        array_compare(x.se2, np.array([[ 0., -3.,  1.],
                                       [ 3.,  0.,  2.],
                                       [ 0.,  0.,  0.]]))

    def test_list_constuctor(self):
        x = Twist2([1, 0, 0])
        
        a = Twist2([x,x,x,x])
        self.assertIsInstance(a, Twist2)
        self.assertEqual(len(a), 4)
        
        a = Twist2([x.se2, x.se2, x.se2, x.se2])
        self.assertIsInstance(a, Twist2)
        self.assertEqual(len(a), 4)
        
        a = Twist2([x.S, x.S, x.S, x.S])
        self.assertIsInstance(a, Twist2)
        self.assertEqual(len(a), 4)
        
        s = [1, 2, 3]
        a = Twist2([s, s, s, s])
        self.assertIsInstance(a, Twist2)
        self.assertEqual(len(a), 4)
        
    def test_predicate(self):
        x = Twist2.R([1, 2])
        self.assertFalse(x.isprismatic)
        
        # check prismatic twist
        x = Twist2.P([1, 2])
        self.assertTrue(x.isprismatic)
        
        self.assertTrue(Twist2.isvalid(x.se2))
        self.assertTrue(Twist2.isvalid(x.S))
        
        self.assertFalse(Twist2.isvalid(2))
        self.assertFalse(Twist2.isvalid(np.eye(3)))
        
    def test_str(self):
        x = Twist2([1, 2, 3])
        s = str(x)
        self.assertIsInstance(s, str)
        self.assertEqual(len(s), 8)
        self.assertEqual(s.count('\n'), 0)
        
        x.append(x)
        s = str(x)
        self.assertIsInstance(s, str)
        self.assertEqual(len(s), 17)
        self.assertEqual(s.count('\n'), 1)
        

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
        x = Twist2.R([0, 0])
        array_compare(x.exp(pi/2), SE2(0, 0, pi/2))
        
        x = Twist2.R([1, 0])
        array_compare(x.exp(pi/2), SE2(1, -1, pi/2))
        
        x = Twist2.R([1, 2])
        array_compare(x.exp(pi/2), SE2(3, 1, pi/2))

    
    def test_arith(self):
        
        # check overloaded *

        T1 = SE2(1, 2, pi / 2)
        T2 = SE2(4, 5, -pi / 4)
        
        x1 = Twist2(T1)
        x2 = Twist2(T2)

        array_compare( (x1 * x2).exp(), T1 * T2)
        array_compare( (x2 * x1).exp(), T2 * T1)
        
    def test_prod(self):
        # check prod
        T1 = SE2(1, 2, pi / 2)
        T2 = SE2(4, 5, -pi / 4)
        
        x1 = Twist2(T1)
        x2 = Twist2(T2)
        
        x = Twist2([x1, x2])
        array_compare( x.prod().SE2, T1 * T2)

# ---------------------------------------------------------------------------------------#
if __name__ == '__main__':

    unittest.main()
