import numpy.testing as nt
import matplotlib.pyplot as plt
import unittest

"""
we will assume that the primitives rotx,trotx, etc. all work
"""
from math import pi
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


class TestSO2: #(unittest.TestCase):


    def test_constructor(self):
        
        
        # null case
        x = SO2()
        self.assertIsInstance(x, SO2)
        self.assertEqual(len(x), 1)
        array_compare(x.A, np.eye(2,2))
        

        
        
        ## from angle
        
        array_compare(SO2(0).A, np.eye(2))
        array_compare(SO2(pi / 2).A, rot2(pi / 2))
        array_compare(SO2(90, unit='deg').A, rot2(pi / 2))
       
        ## from R
        
        array_compare(SO2(np.eye(2,2)).A, np.eye(2,2))
    
        array_compare(SO2( rot2(pi / 2)).A, rot2(pi / 2))
        array_compare(SO2( rot2(pi)).A, rot2(pi))
           
        
        ## R,T
        array_compare(SO2( np.eye(2)).R, np.eye(2))
       
        array_compare(SO2( rot2(pi / 2)).R, rot2(pi / 2))
        
        
        ## vectorised forms of R
        R = SO2.Empty()
        for theta in [-pi / 2, 0, pi / 2, pi]:
            R.append(SO2(theta))
        self.assertEqual(len(R), 4)
        array_compare(R[0], rot2(-pi / 2))
        array_compare(R[3], rot2(pi))

        # TODO self.assertEqual(SO2(R).R, R)
        
        ## copy constructor
        r = SO2(0.3)
        c = SO2(r)
        array_compare(r, c)
        r = SO2(0.4)
        array_compare(c, SO2(0.3))
    
    def test_concat(self):
        x = SO2()
        xx = SO2([x, x, x, x])
        
        self.assertIsInstance(xx, SO2)
        self.assertEqual(len(xx), 4)
    
    def test_primitive_convert(self):
        # char
        
        s = str( SO2())
        self.assertIsInstance(s, str)
    
    
    def test_staticconstructors(self):
        
        ## exponential
        array_compare(SO2.exp( skew(0.3)).R, rot2(0.3))
    
    def test_isa(self):
        
        self.assertTrue(SO2.isvalid(rot2(0)))
    
        self.assertFalse(SO2.isvalid(1))
    
    def test_resulttype(self):
        
        r = SO2()
        self.assertIsInstance(r, SO2)
    
        self.assertIsInstance(r * r, SO2)
        
    
        self.assertIsInstance(r / r, SO2)
        
        self.assertIsInstance(r.inv, SO2)
    
    
    def test_multiply(self):
        
        vx = np.r_[1, 0]
        vy = np.r_[0, 1]
        
        r0 = SO2(0)
        r1 = SO2(pi / 2)
        r2 = SO2(pi)
        u = SO2()
        
        ## SO2-SO2, product
        # scalar x scalar
        
        array_compare(r0 * u, r0)
        array_compare(u * r0, r0)
        
        # vector x vector
        array_compare(SO2([r0, r1, r2]) * SO2([r2, r0, r1]), SO2([r0 * r2, r1 * r0, r2 * r1]))
        
        # scalar x vector
        array_compare(r1 * SO2([r0, r1, r2]), SO2([r1 * r0, r1 * r1, r1 * r2]))
        
        # vector x scalar
        array_compare(SO2([r0, r1, r2]) * r2, SO2([r0 * r2, r1 * r2, r2 * r2]))
        
        ## SO2-vector product
        # scalar x scalar
        
        array_compare(r1 * vx, np.c_[vy])
        
        # vector x vector
        #array_compare(SO2([r0, r1, r0]) * np.c_[vy, vx, vx], np.c_[vy, vy, vx])
        
        # scalar x vector
        array_compare(r1 * np.c_[vx, vy, -vx], np.c_[vy, -vx, -vy])
        
        # vector x scalar
        array_compare(SO2([r0, r1, r2]) * vy, np.c_[vy, -vx, -vy])

    
    def test_divide(self):
        
        r0 = SO2(0)
        r1 = SO2(pi / 2)
        r2 = SO2(pi)
        u = SO2()
        
        # scalar / scalar
        # implicity tests inv
    
        array_compare(r1 / u, r1)
        array_compare(r1 / r1, u)
    
        # vector / vector
        array_compare(SO2([r0, r1, r2]) / SO2([r2, r1, r0]), SO2([r0 / r2, r1 / r1, r2 / r0]))
        
        # vector / scalar
        array_compare(SO2([r0, r1, r2]) / r1, SO2([r0 / r1, r1 / r1, r2 / r1]))
    
    
    def test_conversions(self):
        
        T = SO2(pi / 2).SE2
        self.assertIsInstance(T, SE2)
    
        
        ## Lie stuff
        th = 0.3
        RR = SO2(th)
        array_compare(RR.log(), skew(th))
    
    
    def test_miscellany(self):
        
        r = SO2( 0.3,)
        self.assertAlmostEqual(np.linalg.det(r.A), 1)
        
        self.assertEqual(r.N, 2)
        
        self.assertFalse(r.isSE)
    
    
    def test_display(self):
        
        R = SO2( 0.3,)
        
        R.print()
        
        R.plot()
        
        R2 = SO2(0.6)
        R.animate()
        R.animate(R2)

class TestSE2(unittest.TestCase):
 
    def test_constructor(self):
        
        self.assertIsInstance(SE2(), SE2)
    
        ## null
        array_compare(SE2().A, np.eye(3,3))
        
        # from x,y
        x = SE2(2, 3)
        self.assertIsInstance(x, SE2)
        self.assertEqual(len(x), 1)
        array_compare(x.A, np.array([[1,0,2],[0,1,3],[0,0,1]]))
        
        x = SE2([2, 3])
        self.assertIsInstance(x, SE2)
        self.assertEqual(len(x), 1)
        array_compare(x.A, np.array([[1,0,2],[0,1,3],[0,0,1]]))

        # from x,y,theta
        x = SE2(2, 3, pi / 2)
        self.assertIsInstance(x, SE2)
        self.assertEqual(len(x), 1)
        array_compare(x.A, np.array([[0,-1,2],[1,0,3],[0,0,1]]))
        
        x = SE2([2, 3, pi / 2])
        self.assertIsInstance(x, SE2)
        self.assertEqual(len(x), 1)
        array_compare(x.A, np.array([[0,-1,2],[1,0,3],[0,0,1]]))
        
        x = SE2(2, 3, 90, unit='deg')
        self.assertIsInstance(x, SE2)
        self.assertEqual(len(x), 1)
        array_compare(x.A, np.array([[0,-1,2],[1,0,3],[0,0,1]]))
        
        x = SE2([2, 3, 90], unit='deg')
        self.assertIsInstance(x, SE2)
        self.assertEqual(len(x), 1)
        array_compare(x.A, np.array([[0,-1,2],[1,0,3],[0,0,1]]))
    
        
        ## T
        T = transl2(1, 2) @ trot2(0.3)
        x = SE2(T)
        self.assertIsInstance(x, SE2)
        self.assertEqual(len(x), 1)
        array_compare(x.A, T)
        
        
        ## copy constructor
        TT = SE2(x)
        array_compare(SE2(TT).A, T)
        x = SE2()
        array_compare(SE2(TT).A, T)
        
        ## vectorised versions
        
        T1 = transl2(1,2) @ trot2(0.3)
        T2 = transl2(1,-2) @ trot2(-0.4)
        
        x =SE2([T1, T2, T1, T2])
        self.assertIsInstance(x, SE2)
        self.assertEqual(len(x), 4)
        array_compare(x[0], T1)
        array_compare(x[1], T2)

            
    def test_concat(self):
        x = SE2()
        xx = SE2([x, x, x, x])
        
        self.assertIsInstance(xx, SE2)
        self.assertEqual(len(xx), 4)
    
    
    def test_staticconstructors(self):
        
        ## exponential
        array_compare(SE2.Exp(np.zeros((3,3))), np.eye(3,3))
        t = [1, 2]
        array_compare(SE2.Exp(skewa(np.r_[t, 0])), transl2(t))
    
    
    
    def test_isa(self):
        
        self.assertTrue(SE2.isvalid(trot2(0)))
        self.assertFalse(SE2.isvalid(1))

    
    def test_resulttype(self):
        
        t = SE2()
        self.assertIsInstance(t, SE2)
        self.assertIsInstance(t * t, SE2)
        self.assertIsInstance(t / t, SE2)
        self.assertIsInstance(t.inv, SE2)
        self.assertIsInstance(t + t, np.ndarray)
        self.assertIsInstance(t + 1, np.ndarray)
        self.assertIsInstance(t - 1, np.ndarray)
        self.assertIsInstance(1 + t, np.ndarray)
        self.assertIsInstance(1 - t, np.ndarray)
        self.assertIsInstance(2 * t, np.ndarray)
        self.assertIsInstance(t * 2, np.ndarray)

    
    def test_inverse(self):    
        
        T1 = transl2(1, 2) @ trot2(0.3)
        TT1 = SE2(T1)
        
        # test inverse
        array_compare(TT1.inv.A, np.linalg.inv(T1))
        
        array_compare(TT1 * TT1.inv,  np.eye(3))
        array_compare(TT1.inv * TT1, np.eye(3))
        
        # vector case
        TT2 = SE2([TT1, TT1])
        u = [np.eye(3), np.eye(3)]
        array_compare(TT2.inv * TT1, u)
    
    
    def test_Rt(self):
       
        
        TT1 = SE2.Rand()
        T1 = TT1.A
        R1 = t2r(T1)
        t1 = transl2(T1)
        
        array_compare(TT1.A, T1)
        array_compare(TT1.R, R1)
        array_compare(TT1.t, t1)
        
        TT = SE2([TT1, TT1, TT1])
        array_compare(TT.t, [t1, t1, t1])
    
    
    def test_arith(self):
        
        
        TT1 = SE2.Rand()
        T1 = TT1.A
        TT2 = SE2.Rand()
        T2 = TT2.A
    
        I = SE2()
        
        ## SE2, * SE2, product
        # scalar x scalar
        
        array_compare(TT1 * TT2, T1 @ T2)
        array_compare(TT2 * TT1, T2 @ T1)
        array_compare(TT1 * I, T1)
        array_compare(TT2 * I, TT2)

        
        # vector x vector
        array_compare(SE2([TT1, TT1, TT2]) * SE2([TT2, TT1, TT1]), SE2([TT1*TT2, TT1*TT1, TT2*TT1]))
        
        # scalar x vector
        array_compare(TT1 * SE2([TT2, TT1]), SE2([TT1*TT2, TT1*TT1]))
        
        # vector x scalar
        array_compare(SE2([TT1, TT2]) * TT2, SE2([TT1*TT2, TT2*TT2]))
        
        ## SE2, * vector product
        vx = np.r_[1, 0]
        vy = np.r_[0, 1]
    
        # scalar x scalar
        
        array_compare(TT1 * vy, h2e( T1 @ e2h(vy)))
        
        # # vector x vector
        # array_compare(SE2([TT1, TT2]) * np.c_[vx, vy], np.c_[h2e(T1 @ e2h(vx)), h2e(T2 @ e2h(vy))])
        
        # scalar x vector
        array_compare(TT1 * np.c_[vx, vy], h2e( T1 @ e2h(np.c_[vx, vy])))
        
        # vector x scalar
        array_compare(SE2([TT1, TT2, TT1]) * vy, np.c_[h2e(T1 @ e2h(vy)), h2e(T2 @ e2h(vy)), h2e(T1 @ e2h(vy))])
    
    def test_defs(self):
        
        # log
        x = SE2.Exp([2, 3, 0.5])
        array_compare(x.log, np.array([[0, -0.5, 2], [0.5, 0, 3], [0, 0, 0]]))
    
    
    def test_conversions(self):
        
        
        ##  SE2,                     convert to SE2, class
    
        TT = SE2(1, 2, 0.3)
        
        array_compare(TT, transl2(1, 2) @ trot2(0.3))
        
        ## xyt
        array_compare(TT.xyt, np.r_[1, 2, 0.3])
        
        ## Lie stuff
        th = 0.3
        RR = SO2(th)
        array_compare(RR.log, skew(th))
    
    def test_interp(self):
        TT = SE2(2, -4, 0.6)
        I = SE2()
        
        z = I.interp(TT, 0)
        self.assertIsInstance(z, SE2)
        
        array_compare(I.interp(TT, 0), I)
        array_compare(I.interp(TT, 1), TT)
        array_compare(I.interp(TT, 0.5), SE2(1, -2, 0.3))
    
    def test_miscellany(self):
        
        TT = SE2(1, 2, 0.3)
        
        self.assertEqual(TT.A.shape, (3,3))
            
        self.assertTrue(TT.isSE)
        
        self.assertIsInstance(TT, SE2)
    
    def test_display(self):
        
        T1 = SE2.Rand()
        
        T1.print()
        T1.printline()
        
    def test_graphics(self):
        
        plt.close('all')
        T1 = SE2.Rand()
        T2 = SE2.Rand()
        
        T1.plot(dims=[-2,2])
        
        T1.animate(repeat=False, dims=[-2,2])
        T1.animate(T0=T2, repeat=False, dims=[-2,2])

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