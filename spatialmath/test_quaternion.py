import numpy.testing as nt
import unittest
    
from spatialmath import *
from spatialmath.base import *

import numpy as np
from math import pi, sin, cos

def qcompare(x, y):
    if isinstance(x, Quaternion):
        x = x.vec
    if isinstance(y, Quaternion):
        y = y.vec
    nt.assert_array_almost_equal(x, y)

# straight port of the MATLAB unit tests
            
class TestUnitQuaternion(unittest.TestCase):
    

    def test_constructor(self):
        nt.assert_array_almost_equal(UnitQuaternion().vec, np.r_[1,0,0,0])
        
        nt.assert_array_almost_equal(UnitQuaternion.Rx(90,'deg').vec, np.r_[1,1,0,0]/math.math.sqrt(2))
        nt.assert_array_almost_equal(UnitQuaternion.Rx(-90,'deg').vec, np.r_[1,-1,0,0]/math.math.sqrt(2))
        nt.assert_array_almost_equal(UnitQuaternion.Ry(90,'deg').vec, np.r_[1,0,1,0]/math.math.sqrt(2))
        nt.assert_array_almost_equal(UnitQuaternion.Ry(-90,'deg').vec, np.r_[1,0,-1,0]/math.math.sqrt(2))
        nt.assert_array_almost_equal(UnitQuaternion.Rz(90,'deg').vec, np.r_[1,0,0,1]/math.math.sqrt(2))
        nt.assert_array_almost_equal(UnitQuaternion.Rz(-90,'deg').vec, np.r_[1,0,0,-1]/math.math.sqrt(2))


    def test_constructor(self):
        
        qcompare(UnitQuaternion(), [1, 0, 0, 0])
        
        # from S
        qcompare(UnitQuaternion([1, 0, 0, 0]), np.r_[1, 0, 0, 0])
        qcompare(UnitQuaternion([0, 1, 0, 0]), np.r_[0, 1, 0, 0])
        qcompare(UnitQuaternion([0, 0, 1, 0]), np.r_[0, 0, 1, 0])
        qcompare(UnitQuaternion([0, 0, 0, 1]), np.r_[0, 0, 0, 1])
     
        qcompare(UnitQuaternion([2, 0, 0, 0]), np.r_[1, 0, 0, 0])
        qcompare(UnitQuaternion([-2, 0, 0, 0]), np.r_[-1, 0, 0, 0])
    
        # from [S,V]
        qcompare(UnitQuaternion(1, [0, 0, 0]), np.r_[1, 0, 0, 0])
        qcompare(UnitQuaternion(0, [1, 0, 0]), np.r_[0, 1, 0, 0])
        qcompare(UnitQuaternion(0, [0, 1, 0]), np.r_[0, 0, 1, 0])
        qcompare(UnitQuaternion(0, [0, 0, 1]), np.r_[0, 0, 0, 1])
     
        qcompare(UnitQuaternion(2, [0, 0, 0]), np.r_[1, 0, 0, 0])
        qcompare(UnitQuaternion(-2, [0, 0, 0]), np.r_[-1, 0, 0, 0])
        
        # from R
        
        qcompare(UnitQuaternion( np.eye(3) ), [1, 0, 0, 0])
    
        qcompare(UnitQuaternion( rotx(pi/2) ), np.r_[1, 1, 0, 0]/math.sqrt(2))
        qcompare(UnitQuaternion( roty(pi/2) ), np.r_[1, 0, 1, 0]/math.sqrt(2))
        qcompare(UnitQuaternion( rotz(pi/2) ), np.r_[1, 0, 0, 1]/math.sqrt(2))
        
        qcompare(UnitQuaternion( rotx(-pi/2) ), np.r_[1, -1, 0, 0]/math.sqrt(2))
        qcompare(UnitQuaternion( roty(-pi/2) ), np.r_[1, 0, -1, 0]/math.sqrt(2))
        qcompare(UnitQuaternion( rotz(-pi/2) ), np.r_[1, 0, 0, -1]/math.sqrt(2))
        
        qcompare(UnitQuaternion( rotx(pi) ), np.r_[0, 1, 0, 0])
        qcompare(UnitQuaternion( roty(pi) ), np.r_[0, 0, 1, 0])
        qcompare(UnitQuaternion( rotz(pi) ), np.r_[0, 0, 0, 1])
        
        # from SO3
        
        qcompare(UnitQuaternion( SO3() ), np.r_[1, 0, 0, 0])
    
        qcompare(UnitQuaternion( SO3.Rx(pi/2) ), np.r_[1, 1, 0, 0]/math.sqrt(2))
        qcompare(UnitQuaternion( SO3.Ry(pi/2) ), np.r_[1, 0, 1, 0]/math.sqrt(2))
        qcompare(UnitQuaternion( SO3.Rz(pi/2) ), np.r_[1, 0, 0, 1]/math.sqrt(2))
        
        qcompare(UnitQuaternion( SO3.Rx(-pi/2) ), np.r_[1, -1, 0, 0]/math.sqrt(2))
        qcompare(UnitQuaternion( SO3.Ry(-pi/2) ), np.r_[1, 0, -1, 0]/math.sqrt(2))
        qcompare(UnitQuaternion( SO3.Rz(-pi/2) ), np.r_[1, 0, 0, -1]/math.sqrt(2))
        
        qcompare(UnitQuaternion( SO3.Rx(pi) ), np.r_[0, 1, 0, 0])
        qcompare(UnitQuaternion( SO3.Ry(pi) ), np.r_[0, 0, 1, 0])
        qcompare(UnitQuaternion( SO3.Rz(pi) ), np.r_[0, 0, 0, 1])
        
        # vector of SO3
        qcompare( UnitQuaternion([SO3.Rx(pi/2), SO3.Ry(pi/2), SO3.Rz(pi/2)] ), np.array([[1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1]])/math.sqrt(2))
    
        # from T
        qcompare(UnitQuaternion( trotx(pi/2) ), np.r_[1, 1, 0, 0]/math.sqrt(2))
        qcompare(UnitQuaternion( troty(pi/2) ), np.r_[1, 0, 1, 0]/math.sqrt(2))
        qcompare(UnitQuaternion( trotz(pi/2) ), np.r_[1, 0, 0, 1]/math.sqrt(2))
        
        qcompare(UnitQuaternion( trotx(-pi/2) ), np.r_[1, -1, 0, 0]/math.sqrt(2))
        qcompare(UnitQuaternion( troty(-pi/2) ), np.r_[1, 0, -1, 0]/math.sqrt(2))
        qcompare(UnitQuaternion( trotz(-pi/2) ), np.r_[1, 0, 0, -1]/math.sqrt(2))
        
        qcompare(UnitQuaternion( trotx(pi) ), np.r_[0, 1, 0, 0])
        qcompare(UnitQuaternion( troty(pi) ), np.r_[0, 0, 1, 0])
        qcompare(UnitQuaternion( trotz(pi) ).vec, np.r_[0, 0, 0, 1])
        
        # # vectorised forms of R, T
        # R = []; T = []
        # for theta in [-pi/2, 0, pi/2, pi]:
        #     R = cat(3, R, rotx(theta), roty(theta), rotz(theta))
        #     T = cat(3, T, trotx(theta), troty(theta), trotz(theta))
    
        # nt.assert_array_almost_equal(UnitQuaternion(R).R, R)
        # nt.assert_array_almost_equal(UnitQuaternion(T).T, T)
        
        # copy constructor
        q = UnitQuaternion(rotx(0.3))
        qcompare(UnitQuaternion(q), q)
    
    
    
    def test_concat(self):
        u = UnitQuaternion()
        uu = UnitQuaternion([u, u, u, u])
        
        self.assertIsInstance(uu, UnitQuaternion)
        self.assertEqual(len(uu), 4)
    
    
    def primitive_test_convert(self):
        # char
        
        u = UnitQuaternion()
        
        s = char( u )
        tc.verifyClass(s, 'char')
        s = char( [u, u, u] )
    
        
        # s,v
        nt.assert_array_almost_equal(UnitQuaternion([1, 0, 0, 0]).s, 1)
        nt.assert_array_almost_equal(UnitQuaternion([1, 0, 0, 0]).v, [0, 0, 0])
        
        nt.assert_array_almost_equal(UnitQuaternion([0, 1, 0, 0]).s, 0)
        nt.assert_array_almost_equal(UnitQuaternion([0, 1, 0, 0]).v, [1, 0, 0])
        
        nt.assert_array_almost_equal(UnitQuaternion([0, 0, 1, 0]).s, 0)
        nt.assert_array_almost_equal(UnitQuaternion([0, 0, 1, 0]).v, [0, 1, 0])
        
        nt.assert_array_almost_equal(UnitQuaternion([0, 0, 0, 1]).s, 0)
        nt.assert_array_almost_equal(UnitQuaternion([0, 0, 0, 1]).v, [0, 0, 1])
    
        
        # R,T
        nt.assert_array_almost_equal(u.R, eye(3,3))
       
        nt.assert_array_almost_equal(UnitQuaternion( rotx(pi/2) ).R, rotx(pi/2))
        nt.assert_array_almost_equal(UnitQuaternion( roty(-pi/2) ).R, roty(-pi/2))
        nt.assert_array_almost_equal(UnitQuaternion( rotz(pi) ).R, rotz(pi))
        
        nt.assert_array_almost_equal(UnitQuaternion( rotx(pi/2) ).T, trotx(pi/2))
        nt.assert_array_almost_equal(UnitQuaternion( roty(-pi/2) ).T, troty(-pi/2))
        nt.assert_array_almost_equal(UnitQuaternion( rotz(pi) ).T, trotz(pi))
        
        nt.assert_array_almost_equal(UnitQuaternion.q2r(u.vec), eye(3,3))
    
    
    def test_staticconstructors(self):
        # rotation primitives
        for theta in [-pi/2, 0, pi/2, pi]:
            nt.assert_array_almost_equal(UnitQuaternion.Rx(theta).R, rotx(theta))

        for theta in [-pi/2, 0, pi/2, pi]:
            nt.assert_array_almost_equal(UnitQuaternion.Ry(theta).R, roty(theta))

        for theta in [-pi/2, 0, pi/2, pi]:
            nt.assert_array_almost_equal(UnitQuaternion.Rz(theta).R, rotz(theta))

        
        for theta in np.r_[-pi/2, 0, pi/2, pi]*180/pi:
            nt.assert_array_almost_equal(UnitQuaternion.Rx(theta, 'deg').R, rotx(theta, 'deg'))

        for theta in [-pi/2, 0, pi/2, pi]:
            nt.assert_array_almost_equal(UnitQuaternion.Ry(theta, 'deg').R, roty(theta, 'deg'))

        for theta in [-pi/2, 0, pi/2, pi]:
            nt.assert_array_almost_equal(UnitQuaternion.Rz(theta, 'deg').R, rotz(theta, 'deg'))

        
        # 3 angle
        nt.assert_array_almost_equal(UnitQuaternion.RPY([0.1, 0.2, 0.3] ).R, rpy2r(0.1, 0.2, 0.3 ))
        
        nt.assert_array_almost_equal(UnitQuaternion.Eul([0.1, 0.2, 0.3] ).R, eul2r(0.1, 0.2, 0.3 ))
        
        nt.assert_array_almost_equal(UnitQuaternion.RPY([10, 20, 30], unit='deg' ).R, rpy2r(10, 20, 30, unit='deg' ))
        
        nt.assert_array_almost_equal(UnitQuaternion.Eul([10, 20, 30], unit='deg' ).R, eul2r(10, 20, 30, unit='deg' ))
    
        # (theta, v)
        th = 0.2; v = unitvec([1, 2, 3])
        nt.assert_array_almost_equal(UnitQuaternion.AngVec(th, v ).R, angvec2r(th, v))
        nt.assert_array_almost_equal(UnitQuaternion.AngVec(-th, v ).R, angvec2r(-th, v))
        nt.assert_array_almost_equal(UnitQuaternion.AngVec(-th, -v ).R, angvec2r(-th, -v))
        nt.assert_array_almost_equal(UnitQuaternion.AngVec(th, -v ).R, angvec2r(th, -v))
    
        # (theta, v)
        th = 0.2; v = unitvec([1, 2, 3])
        nt.assert_array_almost_equal(UnitQuaternion.Omega(th*v ).R, angvec2r(th, v))
        nt.assert_array_almost_equal(UnitQuaternion.Omega(-th*v ).R, angvec2r(-th, v))
    
    
    def test_canonic(self):
        R = rotx(0)
        qcompare( UnitQuaternion(R), [1, 0, 0, 0])
        
        R = rotx(pi/2)
        qcompare( UnitQuaternion(R), np.r_[cos(pi/4), sin(pi/4)*np.r_[1, 0, 0]])
        R = roty(pi/2)
        qcompare( UnitQuaternion(R), np.r_[cos(pi/4), sin(pi/4)*np.r_[0, 1, 0]])
        R = rotz(pi/2)
        qcompare( UnitQuaternion(R), np.r_[cos(pi/4), sin(pi/4)*np.r_[0, 0, 1]])
        
        R = rotx(-pi/2)
        qcompare( UnitQuaternion(R), np.r_[cos(pi/4), sin(pi/4)*np.r_[-1, 0, 0]])
        R = roty(-pi/2)
        qcompare( UnitQuaternion(R), np.r_[cos(pi/4), sin(pi/4)*np.r_[0, -1, 0]])
        R = rotz(-pi/2)
        qcompare( UnitQuaternion(R), np.r_[cos(pi/4), sin(pi/4)*np.r_[0, 0, -1]])
        
        R = rotx(pi)
        qcompare( UnitQuaternion(R), np.r_[cos(pi/2), sin(pi/2)*np.r_[1, 0, 0]])
        R = roty(pi)
        qcompare( UnitQuaternion(R), np.r_[cos(pi/2), sin(pi/2)*np.r_[0, 1, 0]])
        R = rotz(pi)
        qcompare( UnitQuaternion(R), np.r_[cos(pi/2), sin(pi/2)*np.r_[0, 0, 1]])
        
        R = rotx(-pi)
        qcompare( UnitQuaternion(R), np.r_[cos(-pi/2), sin(-pi/2)*np.r_[1, 0, 0]])
        R = roty(-pi)
        qcompare( UnitQuaternion(R), np.r_[cos(-pi/2), sin(-pi/2)*np.r_[0, 1, 0]])
        R = rotz(-pi)
        qcompare( UnitQuaternion(R), np.r_[cos(-pi/2), sin(-pi/2)*np.r_[0, 0, 1]])
    
    def test_convert(self):
        # test conversion from rotn matrix to u.quaternion and back
        R = rotx(0)
        qcompare( UnitQuaternion(R).R, R)
        
        R = rotx(pi/2)
        qcompare( UnitQuaternion(R).R, R)
        R = roty(pi/2)
        qcompare( UnitQuaternion(R).R, R)
        R = rotz(pi/2)
        qcompare( UnitQuaternion(R).R, R)
        
        R = rotx(-pi/2)
        qcompare( UnitQuaternion(R).R, R)
        R = roty(-pi/2)
        qcompare( UnitQuaternion(R).R, R)
        R = rotz(-pi/2)
        qcompare( UnitQuaternion(R).R, R)
        
        R = rotx(pi)
        qcompare( UnitQuaternion(R).R, R)
        R = roty(pi)
        qcompare( UnitQuaternion(R).R, R)
        R = rotz(pi)
        qcompare( UnitQuaternion(R).R, R)
        
        R = rotx(-pi)
        qcompare( UnitQuaternion(R).R, R)
        R = roty(-pi)
        qcompare( UnitQuaternion(R).R, R)
        R = rotz(-pi)
        qcompare( UnitQuaternion(R).R, R)
    
    
    def test_resulttype(self):
        
        q = Quaternion([2, 0, 0, 0])
        u = UnitQuaternion()
        
        self.assertIsInstance(q*q, Quaternion)
        self.assertIsInstance(q*u, Quaternion)
        self.assertIsInstance(u*q, Quaternion)
        self.assertIsInstance(u*u, UnitQuaternion)
        
        #self.assertIsInstance(u.*u, UnitQuaternion)
        # other combos all fail, test this?
        
        self.assertIsInstance(u/u, UnitQuaternion)
        
        self.assertIsInstance(u.conj, UnitQuaternion)
        self.assertIsInstance(u.inv, UnitQuaternion)
        self.assertIsInstance(u.unit, UnitQuaternion)
        self.assertIsInstance(q.unit, UnitQuaternion)
        
        self.assertIsInstance(q.conj, Quaternion)
        
        self.assertIsInstance(q+q, Quaternion)
        self.assertIsInstance(q-q, Quaternion)
        
        self.assertIsInstance(u+u, Quaternion)
        self.assertIsInstance(u-u, Quaternion)
        
        # self.assertIsInstance(q+u, Quaternion)
        # self.assertIsInstance(u+q, Quaternion)
        
        # self.assertIsInstance(q-u, Quaternion)
        # self.assertIsInstance(u-q, Quaternion)
        # TODO test for ValueError in these cases
        
        self.assertIsInstance(u.SO3, SO3)
        self.assertIsInstance(u.SE3, SE3)

    
    
    def test_multiply(self):
        
        vx = np.r_[1, 0, 0]; vy = np.r_[0, 1, 0]; vz = np.r_[0, 0, 1]
        rx = UnitQuaternion.Rx(pi/2)
        ry = UnitQuaternion.Ry(pi/2)
        rz = UnitQuaternion.Rz(pi/2)
        u = UnitQuaternion()
        
        # quat-quat product
        # scalar x scalar
        
        qcompare(rx*u, rx)
        qcompare(u*rx, rx); 
        
        #vector x vector
        qcompare(UnitQuaternion([ry, rz, rx]) * UnitQuaternion([rx, ry, rz]), UnitQuaternion([ry*rx, rz*ry, rx*rz]))
        
        # scalar x vector
        qcompare(ry * UnitQuaternion([rx, ry, rz]), UnitQuaternion([ry*rx, ry*ry, ry*rz]))
        
        #vector x scalar
        qcompare(UnitQuaternion([rx, ry, rz]) * ry, UnitQuaternion([rx*ry, ry*ry, rz*ry]))
        
        # quatvector product
        # scalar x scalar
        
        qcompare(rx*vy, vz)
        
        
        # scalar x vector
        nt.assert_array_almost_equal(ry * np.c_[vx, vy, vz], np.c_[-vz, vy, vx])
        
        #vector x scalar
        nt.assert_array_almost_equal(UnitQuaternion([ry, rz, rx]) * vy, np.c_[vy, -vx, vz])
    
    # def multiply_test_normalized(self):
        
    #     vx = [1, 0, 0]; vy = [0, 1, 0]; vz = [0, 0, 1]
    #     rx = UnitQuaternion.Rx(pi/2)
    #     ry = UnitQuaternion.Ry(pi/2)
    #     rz = UnitQuaternion.Rz(pi/2)
    #     u = UnitQuaternion()
        
    #     # quat-quat product
    #     # scalar x scalar
        
    #     nt.assert_array_almost_equal(double(rx.*u), double(rx))
    #     nt.assert_array_almost_equal(double(u.*rx), double(rx))
        
    #     # shouldn't make that much difference here
    #     nt.assert_array_almost_equal(double(rx.*ry), double(rx*ry))
    #     nt.assert_array_almost_equal(double(rx.*rz), double(rx*rz))
        
    #     #vector x vector
    #     #nt.assert_array_almost_equal([ry, rz, rx] .* [rx, ry, rz], [ry.*rx, rz.*ry, rx.*rz])
        
    #     # scalar x vector
    #     #nt.assert_array_almost_equal(ry .* [rx, ry, rz], [ry.*rx, ry.*ry, ry.*rz])
        
    #     #vector x scalar
    #     #nt.assert_array_almost_equal([rx, ry, rz] .* ry, [rx.*ry, ry.*ry, rz.*ry])
    
    def test_divide(self):
        
        rx = UnitQuaternion.Rx(pi/2)
        ry = UnitQuaternion.Ry(pi/2)
        rz = UnitQuaternion.Rz(pi/2)
        u = UnitQuaternion()
        
        # scalar / scalar
        # implicity tests inv
    
        qcompare(rx/u, rx)
        qcompare(ry/ry, u)
    
        #vector /vector
        qcompare(UnitQuaternion([ry, rz, rx]) / UnitQuaternion([rx, ry, rz]), UnitQuaternion([ry/rx, rz/ry, rx/rz]))
        
        #vector / scalar
        qcompare(UnitQuaternion([rx, ry, rz]) / ry, UnitQuaternion([rx/ry, ry/ry, rz/ry]))
        
        # scalar /vector
        qcompare(ry / UnitQuaternion([rx, ry, rz]),  UnitQuaternion([ry/rx, ry/ry, ry/rz]))
    
    
    
    def test_angle(self):
            # angle between quaternions
        # pure
        v = [5, 6, 7]
    
    
    def test_conversions(self):
        
        #, 3 angle
        qcompare(UnitQuaternion.RPY([0.1, 0.2, 0.3] ).rpy, [0.1, 0.2, 0.3])
        
        qcompare(UnitQuaternion.Eul([0.1, 0.2, 0.3] ).eul, [0.1, 0.2, 0.3 ])
        
        qcompare(UnitQuaternion.RPY([10, 20, 30], unit='deg' ).R, rpy2r(10, 20, 30, unit='deg' ))
        
        qcompare(UnitQuaternion.Eul([10, 20, 30], unit='deg' ).R, eul2r(10, 20, 30, unit='deg' ))
        
        # (theta, v)
        th = 0.2; v = unitvec([1, 2, 3])
        [a,b] = UnitQuaternion.AngVec(th, v ).angvec
        self.assertAlmostEqual(a, th)
        nt.assert_array_almost_equal(b, v)
        
        [a,b] = UnitQuaternion.AngVec(-th, v ).angvec
        self.assertAlmostEqual(a, th)
        nt.assert_array_almost_equal(b, -v)
        
        # null rotation case
        th = 0; v = unitvec([1, 2, 3])
        [a,b] = UnitQuaternion.AngVec(th, v ).angvec
        self.assertAlmostEqual(a, th)
        
    
    #  SO3                     convert to SO3 class
    #  SE3                     convert to SE3 class
    
    
    def test_miscellany(self):
        
        # AbsTol not used since Quaternion supports eq() operator
        
        rx = UnitQuaternion.Rx(pi/2)
        ry = UnitQuaternion.Ry(pi/2)
        rz = UnitQuaternion.Rz(pi/2)
        u = UnitQuaternion()
        
        # norm
        qcompare(rx.norm, 1)
        qcompare(UnitQuaternion([rx, ry, rz]).norm, [1, 1, 1])
        
        # unit
        qcompare(rx.unit, rx)
        qcompare(UnitQuaternion([rx, ry, rz]).unit, UnitQuaternion([rx, ry, rz]))
        
        # inner
        nt.assert_array_almost_equal(u.inner(u), 1)
        nt.assert_array_almost_equal(rx.inner(ry), 0.5)
        nt.assert_array_almost_equal(rz.inner(rz), 1)
    
    
        q = rx*ry*rz
            
        qcompare(q**0, u)
        qcompare(q**(-1), q.inv)
        qcompare(q**2, q*q)
        
        # angle
        # self.assertEqual(angle(u, u), 0)
        # self.assertEqual(angle(u, rx), pi/4)
        # self.assertEqual(angle(u, [rx, u]), pi/4*np.r_[1, 0])
        # self.assertEqual(angle([rx, u], u), pi/4*np.r_[1, 0])
        # self.assertEqual(angle([rx, u], [u, rx]), pi/4*np.r_[1, 1])
        # TODO angle
    
        
        # increment
        # w = [0.02, 0.03, 0.04]
        
        # nt.assert_array_almost_equal(rx.increment(w), rx*UnitQuaternion.omega(w))
    
    def test_interp(self):
            
        rx = UnitQuaternion.Rx(pi/2)
        ry = UnitQuaternion.Ry(pi/2)
        rz = UnitQuaternion.Rz(pi/2)
        u = UnitQuaternion()
        
        q = rx*ry*rz
        
        # from null
        qcompare(q.interp(0), u)
        qcompare(q.interp(1), q )
        
        #self.assertEqual(length(q.interp(linspace(0,1, 10))), 10)
        #self.assertTrue(all( q.interp([0, 1]) == [u, q]))
        # TODO vectorizing
        
        q0_5 = q.interp(0.5)
        qcompare( q0_5 * q0_5, q)
        
        # between two quaternions
        qcompare(q.interp(0, dest=rx), q )
        qcompare(q.interp(1, dest=rx), rx )
        
        #self.assertTrue(all( q.interp([0, 1], dest=rx, ) == [q, rx]))
        
        # test shortest option
        # q1 = UnitQuaternion.Rx(0.9*pi)
        # q2 = UnitQuaternion.Rx(-0.9*pi)
        # qq = q1.interp(q2, 11)
        # qcompare( qq(6), UnitQuaternion.Rx(0) )
        # qq = q1.interp(q2, 11, 'shortest')
        # qcompare( qq(6), UnitQuaternion.Rx(pi) )
        # TODO interp
    
    
    def test_eq(self):
        q1 = UnitQuaternion([0, 1, 0, 0])
        q2 = UnitQuaternion([0, -1, 0, 0])
        q3 = UnitQuaternion.Rz(pi/2)
        
        self.assertTrue( q1 == q1)
        self.assertTrue( q2 == q2)
        self.assertTrue( q3 == q3)
        self.assertTrue( q1 == q2)  # because of double wrapping
        self.assertFalse( q1 == q3)
        
        nt.assert_array_almost_equal( [q1, q1, q1] == [q1, q1, q1], [True, True, True])
        nt.assert_array_almost_equal( [q1, q2, q3] == [q1, q2, q3], [True, True, True])
        nt.assert_array_almost_equal( [q1, q1, q3] == q1, [True, True, False])
        nt.assert_array_almost_equal( q3 == [q1, q1, q3], [False, False, True])
    
    
    
    def test_logical(self):
        rx = UnitQuaternion.Rx(pi/2)
        ry = UnitQuaternion.Ry(pi/2)
        
        # equality tests
        self.assertTrue(rx == rx)
        self.assertFalse(rx != rx)
        self.assertFalse(rx == ry)
    
    
    
    def test_dot(self):
        q = UnitQuaternion()
        omega = np.r_[1, 2, 3]
        
        nt.assert_array_almost_equal(q.dot(omega), np.r_[0, omega/2])
        nt.assert_array_almost_equal(q.dotb(omega), np.r_[0, omega/2])
        
        q = UnitQuaternion.Rx(pi/2)
        qcompare(q.dot(omega), 0.5*Quaternion.pure(omega)*q)
        qcompare(q.dotb(omega), 0.5*q*Quaternion.pure(omega))
    
    def test_matrix(self):
        
        q1 = UnitQuaternion.RPY([0.1, 0.2, 0.3])
        q2 = UnitQuaternion.RPY([0.2, 0.3, 0.4])
        
        qcompare(q1*q2, q1.matrix @ q2.vec)
    
    
    def test_vec3(self):
        
        q1 = UnitQuaternion.RPY([0.1, 0.2, 0.3])
        q2 = UnitQuaternion.RPY([0.2, 0.3, 0.4])
        
        q12 = q1 * q2
        
        q1v = q1.vec3; q2v = q2.vec3
        
        q12v = UnitQuaternion.qvmul(q1v, q2v)
        
        q12_ = UnitQuaternion.Vec3(q12v)
        
        qcompare(q12, q12_)
    
    
     
    # def test_display(self):
    #     ry = UnitQuaternion.Ry(pi/2)

    #     ry.plot()
    #     h = ry.plot()
    #     ry.animate()
    #     ry.animate('rgb')
    #     ry.animate( UnitQuaternion.Rx(pi/2), 'rgb' )


        
class TestQuaternion(unittest.TestCase):
        
    def test_constructor(self):
    
        q = Quaternion()
        self.assertEqual(len(q),  1)
        self.assertIsInstance(q, Quaternion)
        
        nt.assert_array_almost_equal(Quaternion().vec, [0, 0, 0, 0])
        
        # from S
        nt.assert_array_almost_equal(Quaternion([1, 0, 0, 0]).vec, [1, 0, 0, 0])
        nt.assert_array_almost_equal(Quaternion([0, 1, 0, 0]).vec, [0, 1, 0, 0])
        nt.assert_array_almost_equal(Quaternion([0, 0, 1, 0]).vec, [0, 0, 1, 0])
        nt.assert_array_almost_equal(Quaternion([0, 0, 0, 1]).vec, [0, 0, 0, 1])
        
        nt.assert_array_almost_equal(Quaternion([2, 0, 0, 0]).vec, [2, 0, 0, 0])
        nt.assert_array_almost_equal(Quaternion([-2, 0, 0, 0]).vec, [-2, 0, 0, 0])
        
        # from [S,V]
        nt.assert_array_almost_equal(Quaternion(1, [0, 0, 0]).vec, [1, 0, 0, 0])
        nt.assert_array_almost_equal(Quaternion(0, [1, 0, 0]).vec, [0, 1, 0, 0])
        nt.assert_array_almost_equal(Quaternion(0, [0, 1, 0]).vec, [0, 0, 1, 0])
        nt.assert_array_almost_equal(Quaternion(0, [0, 0, 1]).vec, [0, 0, 0, 1])
        
        nt.assert_array_almost_equal(Quaternion(2, [0, 0, 0]).vec, [2, 0, 0, 0])
        nt.assert_array_almost_equal(Quaternion(-2, [0, 0, 0]).vec, [-2, 0, 0, 0])
        
        # pure
        v = [5, 6, 7]
        nt.assert_array_almost_equal(Quaternion.pure(v).vec, [0,]+v)
        
        ##tc.verifyError( @() Quaternion.pure([1, 2]), 'SMTB:Quaternion:badarg')
        
        # copy constructor
        q = Quaternion([1, 2, 3, 4])
        nt.assert_array_almost_equal(Quaternion(q).vec, q.vec)
        
        
        # errors
        
        ##tc.verifyError( @() Quaternion(2), 'SMTB:Quaternion:badarg')
        ##tc.verifyError( @() Quaternion([1, 2, 3]), 'SMTB:Quaternion:badarg')
    
    def test_properties(self):
        
        q = Quaternion([1,2,3,4])
        self.assertEqual(q.s, 1)
        nt.assert_array_almost_equal(q.v, np.r_[2,3,4])
        nt.assert_array_almost_equal(q.vec, np.r_[1,2,3,4])
    
    
    def log_test_exp(self):
        
        q1 = Quaternion([4, 3, 2, 1])
        q2 = Quaternion([-1, 2, -3, 4])
        
        nt.assert_array_almost_equal(exp(log(q1)), q1)
        nt.assert_array_almost_equal(exp(log(q2)), q2)
        
        #nt.assert_array_almost_equal(log(exp(q1)), q1)
        #nt.assert_array_almost_equal(log(exp(q2)), q2)
    
    
    def test_char(self):
        
        # char
        q = Quaternion()
        
        s = str( q )
        self.assertTrue(isinstance(s, str))
        self.assertEqual(len(s), 42)
        
        s = str( Quaternion([q, q, q]) )
        self.assertEqual(len(s), 126)
        self.assertEqual(s.count('\n'), 3)
        
        
        # symbolic display
        # syms s x y z real
        # q = Quaternion([s x y z])
        # s = char( q )
        # tc.verifyTrue( ischar(s))
        # nt.assert_array_almost_equal(size(s,1), 1)
        
        # s = char( [q q q] )
        # tc.verifyTrue( ischar(s))
        # nt.assert_array_almost_equal(size(s,1), 3)
        
    
    
    def test_concat(self):
        u = Quaternion()
        uu = Quaternion([u, u, u, u])
        
        self.assertIsInstance(uu, Quaternion)
        self.assertEqual(len(uu),  4)        
    
    
    def primitive_test_convert(self):
        
        
        # s,v
        nt.assert_array_almost_equal(Quaternion([1, 0, 0, 0]).s, 1)
        nt.assert_array_almost_equal(Quaternion([1, 0, 0, 0]).v, [0, 0, 0])
        
        nt.assert_array_almost_equal(Quaternion([0, 1, 0, 0]).s, 0)
        nt.assert_array_almost_equal(Quaternion([0, 1, 0, 0]).v, [1, 0, 0])
        
        nt.assert_array_almost_equal(Quaternion([0, 0, 1, 0]).s, 0)
        nt.assert_array_almost_equal(Quaternion([0, 0, 1, 0]).v, [0, 1, 0])
        
        nt.assert_array_almost_equal(Quaternion([0, 0, 0, 1]).s, 0)
        nt.assert_array_almost_equal(Quaternion([0, 0, 0, 1]).v, [0, 0, 1])
        
    
    def test_resulttype(self):
        
        q = Quaternion([2, 0, 0, 0])
        
        self.assertIsInstance(q, Quaternion)

        
        # other combos all fail, test this?
                
        self.assertIsInstance(q.conj, Quaternion)
        self.assertIsInstance(q.unit, UnitQuaternion)
        
        self.assertIsInstance(q+q, Quaternion)
        self.assertIsInstance(q+q, Quaternion)
    
    
    def test_multiply(self):
        
        q1 = Quaternion([1, 2, 3, 4])
        q2 = Quaternion([4, 3, 2, 1])
        q3 = Quaternion([-1, 2, -3, 4])
        
        u = Quaternion([1, 0, 0, 0])
    
        
        # quat-quat product
        # scalar x scalar
        
        qcompare(q1*u, q1)
        qcompare(u*q1, q1)
        qcompare(q1*q2, [-12, 6, 24, 12])
        
        q = q1
        q *= q2
        qcompare(q, [-12, 6, 24, 12])
                
        
        #vector x vector
        qcompare(Quaternion([q1, u, q2, u, q3, u]) * Quaternion([u, q1, u, q2, u, q3]), Quaternion([q1, q1, q2, q2, q3, q3]))
        
        q = Quaternion([q1, u, q2, u, q3, u])
        q *= Quaternion([u, q1, u, q2, u, q3])
        qcompare(q, Quaternion([q1, q1, q2, q2, q3, q3]))
        
        # scalar x vector
        qcompare(q1 * Quaternion([q1, q2, q3]), Quaternion([q1*q1, q1*q2, q1*q3]))
        
        #vector x scalar
        qcompare(Quaternion([q1, q2, q3]) * q2, Quaternion([q1*q2, q2*q2, q3*q2]))
        
        # quat-real product
        # scalar x scalar
        
        v1 = q1.vec
        qcompare(q1*5, v1*5)
        qcompare(6*q1, v1*6)
        qcompare(-2*q1, -2*v1)
        
        # scalar x vector
        qcompare(5*Quaternion([q1, q2, q3]), Quaternion([5*q1, 5*q2, 5*q3]))
        
        #vector x scalar
        qcompare(Quaternion([q1, q2, q3]) * 5, Quaternion([5*q1, 5*q2, 5*q3]))
        
        # matrix form of multiplication
        qcompare(q1.matrix @ q2.vec, q1*q2 )
        
        # quat-scalar product
        qcompare(q1*2, q1.vec*2)
        qcompare(Quaternion([q1*2, q2*2]), Quaternion([q1, q2])*2)
        
        # errors
        
        # tc.verifyError( @() q1 * [1, 2, 3], 'SMTB:Quaternion:badarg')
        # tc.verifyError( @() [1, 2, 3]*q1, 'SMTB:Quaternion:badarg')
        # tc.verifyError( @() [q1, q1] * [q1, q1, q1], 'SMTB:Quaternion:badarg')
        # tc.verifyError( @() q1*SE3, 'SMTB:Quaternion:badarg')
    
    
    
    def test_equality(self):
        q1 = Quaternion([1, 2, 3, 4])
        q2 = Quaternion([-2, 1, -4, 3])
        
        self.assertTrue(q1 == q1)
        self.assertFalse(q1 == q2)
        
        self.assertTrue(q1 != q2)
        self.assertFalse(q2 != q2)
        
        qt1 = Quaternion([q1, q1, q2, q2])
        qt2 = Quaternion([q1, q2, q2, q1])
        
        self.assertEqual(qt1==q1, [True, True, False, False])
        self.assertEqual(q1==qt1, [True, True, False, False])
        self.assertEqual(qt1==qt1, [True, True, True, True])
        
        self.assertEqual(qt2==q1, [True, False, False, True])
        self.assertEqual(q1==qt2, [True, False, False, True])
        self.assertEqual(qt1==qt2, [True, False, True, False])
        
        self.assertEqual(qt1!=q1, [False, False, True, True])
        self.assertEqual(q1!=qt1, [False, False, True, True])
        self.assertEqual(qt1!=qt1, [False, False, False, False])
        
        self.assertEqual(qt2!=q1, [False, True, True, False])
        self.assertEqual(q1!=qt2, [False, True, True, False])
        self.assertEqual(qt1!=qt2, [False, True, False, True])
        
        # errors
        
        # tc.verifyError( @() [q1 q1] == [q1 q1 q1], 'SMTB:Quaternion:badarg')
        # tc.verifyError( @() [q1 q1] != [q1 q1 q1], 'SMTB:Quaternion:badarg')
    
    
    def basic_test_multiply(self):
        # test run multiplication tests on quaternions
        q = Quaternion([1, 0, 0, 0]) * Quaternion([1, 0, 0, 0])
        qcompare(q.vec, [1, 0, 0, 0 ])
        
        q = Quaternion([1, 0, 0, 0]) * Quaternion([1, 2, 3, 4])
        qcompare(q.vec, [1, 2, 3, 4])
        
        q = Quaternion([1, 2, 3, 4]) * Quaternion([1, 2, 3, 4])
        qcompare(q.vec, [-28, 4, 6, 8])
    
    def add_test_sub(self):
        v1 = [1, 2, 3, 4]; v2 = [2, 2, 4, 7]
        
        # plus
        q = Quaternion(v1) + Quaternion(v2)
        q2 = Quaternion(v1) + v2
        
        qcompare(q.vec, v1+v2)
        qcompare(q2.vec, v1+v2)
        
        # minus
        q = Quaternion(v1) - Quaternion(v2)
        q2 = Quaternion(v1) - v2
        qcompare(q.vec, v1-v2)
        qcompare(q2.vec, v1-v2)
    
    
    
    def test_power(self):
        
        q = Quaternion([1, 2, 3, 4])
        
        qcompare(q**0, Quaternion([1, 0, 0, 0]))
        qcompare(q**1, q)
        qcompare(q**2, q*q)
    
    
    def test_miscellany(self):
        v = np.r_[1, 2, 3, 4]
        q = Quaternion(v)
        u = Quaternion([1, 0, 0, 0])
        
        # norm
        nt.assert_array_almost_equal(q.norm, np.linalg.norm(v))
        nt.assert_array_almost_equal(Quaternion([q, u, q]).norm, [np.linalg.norm(v), 1, np.linalg.norm(v)])
        
        # unit
        qu = q.unit
        u = UnitQuaternion()
        self.assertIsInstance(q, Quaternion)
        nt.assert_array_almost_equal(qu.vec, v/np.linalg.norm(v))
        qcompare(Quaternion([q, u, q]).unit, UnitQuaternion([qu, u ,qu]))
        
        # inner
        nt.assert_equal(u.inner(u), 1)
        nt.assert_equal(q.inner(q), q.norm**2)
        nt.assert_equal(q.inner(u), np.dot(q.vec, u.vec))
 
            
# ---------------------------------------------------------------------------------------#
if __name__ == '__main__':

    
    unittest.main()