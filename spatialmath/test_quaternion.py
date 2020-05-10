import numpy.testing as nt
import unittest
    
from spatialmath.quaternion import *
import numpy as np
from math import pi
import spatialmath.base as tr

def qcompare(x, y):
    if isinstance(x, Quaternion):
        x = x.vec
    if isinstance(y, Quaternion):
        y = y.vec
    nt.assert_array_almost_equal(x, y)
    
            
class TestUnitQuaternion(unittest.TestCase):
    

        
#     def test_constructor(self):
#         nt.assert_array_almost_equal(UnitQuaternion().vec, np.r_[1,0,0,0])
        
#         nt.assert_array_almost_equal(UnitQuaternion.Rx(90,'deg').vec, np.r_[1,1,0,0]/math.sqrt(2))
#         nt.assert_array_almost_equal(UnitQuaternion.Rx(-90,'deg').vec, np.r_[1,-1,0,0]/math.sqrt(2))
#         nt.assert_array_almost_equal(UnitQuaternion.Ry(90,'deg').vec, np.r_[1,0,1,0]/math.sqrt(2))
#         nt.assert_array_almost_equal(UnitQuaternion.Ry(-90,'deg').vec, np.r_[1,0,-1,0]/math.sqrt(2))
#         nt.assert_array_almost_equal(UnitQuaternion.Rz(90,'deg').vec, np.r_[1,0,0,1]/math.sqrt(2))
#         nt.assert_array_almost_equal(UnitQuaternion.Rz(-90,'deg').vec, np.r_[1,0,0,-1]/math.sqrt(2))


#     def test_constructor(self):
        
#         nt.assert_array_almost_equal(UnitQuaternion().vec, [1, 0, 0, 0])
        
#         # from S
#         nt.assert_array_almost_equal(UnitQuaternion([1, 0, 0, 0]).vec, [1, 0, 0, 0])
#         nt.assert_array_almost_equal(UnitQuaternion([0, 1, 0, 0]).vec, [0, 1, 0, 0])
#         nt.assert_array_almost_equal(UnitQuaternion([0, 0, 1, 0]).vec, [0, 0, 1, 0])
#         nt.assert_array_almost_equal(UnitQuaternion([0, 0, 0, 1]).vec, [0, 0, 0, 1])
     
#         nt.assert_array_almost_equal(UnitQuaternion([2, 0, 0, 0]).vec, [1, 0, 0, 0])
#         nt.assert_array_almost_equal(UnitQuaternion([-2, 0, 0, 0]).vec, [-1, 0, 0, 0])
    
#         # from [S,V]
#         nt.assert_array_almost_equal(UnitQuaternion(1, [0, 0, 0]).vec, [1, 0, 0, 0])
#         nt.assert_array_almost_equal(UnitQuaternion(0, [1, 0, 0]).vec, [0, 1, 0, 0])
#         nt.assert_array_almost_equal(UnitQuaternion(0, [0, 1, 0]).vec, [0, 0, 1, 0])
#         nt.assert_array_almost_equal(UnitQuaternion(0, [0, 0, 1]).vec, [0, 0, 0, 1])
     
#         nt.assert_array_almost_equal(UnitQuaternion(2, [0, 0, 0]).vec, [1, 0, 0, 0])
#         nt.assert_array_almost_equal(UnitQuaternion(-2, [0, 0, 0]).vec, [-1, 0, 0, 0])
        
#         # from R
        
#         nt.assert_array_almost_equal(UnitQuaternion( eye(3,3) ).vec, [1, 0, 0, 0])
    
#         nt.assert_array_almost_equal(UnitQuaternion( rotx(pi/2) ).vec, [1, 1, 0, 0]/sqrt(2))
#         nt.assert_array_almost_equal(UnitQuaternion( roty(pi/2) ).vec, [1, 0, 1, 0]/sqrt(2))
#         nt.assert_array_almost_equal(UnitQuaternion( rotz(pi/2) ).vec, [1, 0, 0, 1]/sqrt(2))
        
#         nt.assert_array_almost_equal(UnitQuaternion( rotx(-pi/2) ).vec, [1, -1, 0, 0]/sqrt(2))
#         nt.assert_array_almost_equal(UnitQuaternion( roty(-pi/2) ).vec, [1, 0, -1, 0]/sqrt(2))
#         nt.assert_array_almost_equal(UnitQuaternion( rotz(-pi/2) ).vec, [1, 0, 0, -1]/sqrt(2))
        
#         nt.assert_array_almost_equal(UnitQuaternion( rotx(pi) ).vec, [0, 1, 0, 0])
#         nt.assert_array_almost_equal(UnitQuaternion( roty(pi) ).vec, [0, 0, 1, 0])
#         nt.assert_array_almost_equal(UnitQuaternion( rotz(pi) ).vec, [0, 0, 0, 1])
        
#         # from SO3
        
#         nt.assert_array_almost_equal(UnitQuaternion( SO3 ).vec, [1, 0, 0, 0])
    
#         nt.assert_array_almost_equal(UnitQuaternion( SO3.Rx(pi/2) ).vec, [1, 1, 0, 0]/sqrt(2))
#         nt.assert_array_almost_equal(UnitQuaternion( SO3.Ry(pi/2) ).vec, [1, 0, 1, 0]/sqrt(2))
#         nt.assert_array_almost_equal(UnitQuaternion( SO3.Rz(pi/2) ).vec, [1, 0, 0, 1]/sqrt(2))
        
#         nt.assert_array_almost_equal(UnitQuaternion( SO3.Rx(-pi/2) ).vec, [1, -1, 0, 0]/sqrt(2))
#         nt.assert_array_almost_equal(UnitQuaternion( SO3.Ry(-pi/2) ).vec, [1, 0, -1, 0]/sqrt(2))
#         nt.assert_array_almost_equal(UnitQuaternion( SO3.Rz(-pi/2) ).vec, [1, 0, 0, -1]/sqrt(2))
        
#         nt.assert_array_almost_equal(UnitQuaternion( SO3.Rx(pi) ).vec, [0, 1, 0, 0])
#         nt.assert_array_almost_equal(UnitQuaternion( SO3.Ry(pi) ).vec, [0, 0, 1, 0])
#         nt.assert_array_almost_equal(UnitQuaternion( SO3.Rz(pi) ).vec, [0, 0, 0, 1])
        
#         #vector of SO3
#         nt.assert_array_almost_equal(UnitQuaternion( [SO3.Rx(pi/2) SO3.Ry(pi/2) SO3.Rz(pi/2)] ).vec, [1, 1, 0, 0;, 1, 0, 1, 0;, 1, 0, 0, 1]/sqrt(2))
    
#         # from T
#         nt.assert_array_almost_equal(UnitQuaternion( trotx(pi/2) ).vec, [1, 1, 0, 0]/sqrt(2))
#         nt.assert_array_almost_equal(UnitQuaternion( troty(pi/2) ).vec, [1, 0, 1, 0]/sqrt(2))
#         nt.assert_array_almost_equal(UnitQuaternion( trotz(pi/2) ).vec, [1, 0, 0, 1]/sqrt(2))
        
#         nt.assert_array_almost_equal(UnitQuaternion( trotx(-pi/2) ).vec, [1, -1, 0, 0]/sqrt(2))
#         nt.assert_array_almost_equal(UnitQuaternion( troty(-pi/2) ).vec, [1, 0, -1, 0]/sqrt(2))
#         nt.assert_array_almost_equal(UnitQuaternion( trotz(-pi/2) ).vec, [1, 0, 0, -1]/sqrt(2))
        
#         nt.assert_array_almost_equal(UnitQuaternion( trotx(pi) ).vec, [0, 1, 0, 0])
#         nt.assert_array_almost_equal(UnitQuaternion( troty(pi) ).vec, [0, 0, 1, 0])
#         nt.assert_array_almost_equal(UnitQuaternion( trotz(pi) ).vec, [0, 0, 0, 1])
        
#         #vectorised forms of R, T
#         R = []; T = []
#         for theta = [-pi/2, 0 pi/2 pi]
#             R = cat(3, R, rotx(theta), roty(theta), rotz(theta))
#             T = cat(3, T, trotx(theta), troty(theta), trotz(theta))
    
#         end
#         nt.assert_array_almost_equal(UnitQuaternion(R).R, R)
#         nt.assert_array_almost_equal(UnitQuaternion(T).T, T)
        
#         # copy constructor
#         q = UnitQuaternion(rotx(0.3))
#         nt.assert_array_almost_equal(UnitQuaternion(q), q)
    
    
    
#     def test_concat(self):
#         u = UnitQuaternion()
#         uu = [u u u u]
        
#         tc.verifyClass(uu, 'UnitQuaternion')
#         tc.verifySize(uu, [1, 4])
    
    
#     def primitive_test_convert(self):
#         # char
        
#         u = UnitQuaternion()
        
#         s = char( u )
#         tc.verifyClass(s, 'char')
#         s = char( [u u u] )
    
        
#         # s,v
#         nt.assert_array_almost_equal(UnitQuaternion([1, 0, 0, 0]).s, 1)
#         nt.assert_array_almost_equal(UnitQuaternion([1, 0, 0, 0]).v, [0, 0, 0])
        
#         nt.assert_array_almost_equal(UnitQuaternion([0, 1, 0, 0]).s, 0)
#         nt.assert_array_almost_equal(UnitQuaternion([0, 1, 0, 0]).v, [1, 0, 0])
        
#         nt.assert_array_almost_equal(UnitQuaternion([0, 0, 1, 0]).s, 0)
#         nt.assert_array_almost_equal(UnitQuaternion([0, 0, 1, 0]).v, [0, 1, 0])
        
#         nt.assert_array_almost_equal(UnitQuaternion([0, 0, 0, 1]).s, 0)
#         nt.assert_array_almost_equal(UnitQuaternion([0, 0, 0, 1]).v, [0, 0, 1])
    
        
#         # R,T
#         nt.assert_array_almost_equal(u.R, eye(3,3))
       
#         nt.assert_array_almost_equal(UnitQuaternion( rotx(pi/2) ).R, rotx(pi/2))
#         nt.assert_array_almost_equal(UnitQuaternion( roty(-pi/2) ).R, roty(-pi/2))
#         nt.assert_array_almost_equal(UnitQuaternion( rotz(pi) ).R, rotz(pi))
        
#         nt.assert_array_almost_equal(UnitQuaternion( rotx(pi/2) ).T, trotx(pi/2))
#         nt.assert_array_almost_equal(UnitQuaternion( roty(-pi/2) ).T, troty(-pi/2))
#         nt.assert_array_almost_equal(UnitQuaternion( rotz(pi) ).T, trotz(pi))
        
#         nt.assert_array_almost_equal(UnitQuaternion.q2r(u.vec), eye(3,3))
    
    
#     def test_staticconstructors(self):
#         # rotation primitives
#         for theta = [-pi/2, 0 pi/2 pi]
#             nt.assert_array_almost_equal(UnitQuaternion.Rx(theta).R, rotx(theta))
#         end
#         for theta = [-pi/2, 0 pi/2 pi]
#             nt.assert_array_almost_equal(UnitQuaternion.Ry(theta).R, roty(theta))
#         end
#         for theta = [-pi/2, 0 pi/2 pi]
#             nt.assert_array_almost_equal(UnitQuaternion.Rz(theta).R, rotz(theta))
#         end
        
#             for theta = [-pi/2, 0 pi/2 pi]*180/pi
#             nt.assert_array_almost_equal(UnitQuaternion.Rx(theta, 'deg').R, rotx(theta, 'deg'))
#         end
#         for theta = [-pi/2, 0 pi/2 pi]
#             nt.assert_array_almost_equal(UnitQuaternion.Ry(theta, 'deg').R, roty(theta, 'deg'))
#         end
#         for theta = [-pi/2, 0 pi/2 pi]
#             nt.assert_array_almost_equal(UnitQuaternion.Rz(theta, 'deg').R, rotz(theta, 'deg'))
#         end
        
#         #, 3 angle
#         nt.assert_array_almost_equal(UnitQuaternion.rpy(, 0.1, 0.2, 0.3 ).R, rpy2r(, 0.1, 0.2, 0.3 ))
#         nt.assert_array_almost_equal(UnitQuaternion.rpy([, 0.1, 0.2, 0.3] ).R, rpy2r(, 0.1, 0.2, 0.3 ))
        
#         nt.assert_array_almost_equal(UnitQuaternion.eul(, 0.1, 0.2, 0.3 ).R, eul2r(, 0.1, 0.2, 0.3 ))
#         nt.assert_array_almost_equal(UnitQuaternion.eul([, 0.1, 0.2, 0.3] ).R, eul2r(, 0.1, 0.2, 0.3 ))
        
#         nt.assert_array_almost_equal(UnitQuaternion.rpy(, 10, 20, 30, 'deg' ).R, rpy2r(, 10, 20, 30, 'deg' ))
#         nt.assert_array_almost_equal(UnitQuaternion.rpy([, 10, 20, 30], 'deg' ).R, rpy2r(, 10, 20, 30, 'deg' ))
        
#         nt.assert_array_almost_equal(UnitQuaternion.eul(, 10, 20, 30, 'deg' ).R, eul2r(, 10, 20, 30, 'deg' ))
#         nt.assert_array_almost_equal(UnitQuaternion.eul([, 10, 20, 30], 'deg' ).R, eul2r(, 10, 20, 30, 'deg' ))
    
#         # (theta, v)
#         th =, 0.2; v = unit([1, 2, 3])
#         nt.assert_array_almost_equal(UnitQuaternion.an.vec(th, v ).R, an.vec2r(th, v))
#         nt.assert_array_almost_equal(UnitQuaternion.an.vec(-th, v ).R, an.vec2r(-th, v))
#         nt.assert_array_almost_equal(UnitQuaternion.an.vec(-th, -v ).R, an.vec2r(-th, -v))
#         nt.assert_array_almost_equal(UnitQuaternion.an.vec(th, -v ).R, an.vec2r(th, -v))
    
#         # (theta, v)
#         th =, 0.2; v = unit([1, 2, 3])
#         nt.assert_array_almost_equal(UnitQuaternion.omega(th*v ).R, an.vec2r(th, v))
#         nt.assert_array_almost_equal(UnitQuaternion.omega(-th*v ).R, an.vec2r(-th, v))
    
    
#     def test_canonic(self):
#         R = rotx(0)
#         nt.assert_array_almost_equal( UnitQuaternion(R).vec, [1, 0, 0, 0])
        
#         R = rotx(pi/2)
#         nt.assert_array_almost_equal( UnitQuaternion(R).vec, [cos(pi/4) sin(pi/4)*[1, 0, 0]])
#         R = roty(pi/2)
#         nt.assert_array_almost_equal( UnitQuaternion(R).vec, [cos(pi/4) sin(pi/4)*[0, 1, 0]])
#         R = rotz(pi/2)
#         nt.assert_array_almost_equal( UnitQuaternion(R).vec, [cos(pi/4) sin(pi/4)*[0, 0, 1]])
        
#         R = rotx(-pi/2)
#         nt.assert_array_almost_equal( UnitQuaternion(R).vec, [cos(pi/4) sin(pi/4)*[-1, 0, 0]])
#         R = roty(-pi/2)
#         nt.assert_array_almost_equal( UnitQuaternion(R).vec, [cos(pi/4) sin(pi/4)*[0, -1, 0]])
#         R = rotz(-pi/2)
#         nt.assert_array_almost_equal( UnitQuaternion(R).vec, [cos(pi/4) sin(pi/4)*[0, 0, -1]])
        
#         R = rotx(pi)
#         nt.assert_array_almost_equal( UnitQuaternion(R).vec, [cos(pi/2) sin(pi/2)*[1, 0, 0]])
#         R = roty(pi)
#         nt.assert_array_almost_equal( UnitQuaternion(R).vec, [cos(pi/2) sin(pi/2)*[0, 1, 0]])
#         R = rotz(pi)
#         nt.assert_array_almost_equal( UnitQuaternion(R).vec, [cos(pi/2) sin(pi/2)*[0, 0, 1]])
        
#         R = rotx(-pi)
#         nt.assert_array_almost_equal( UnitQuaternion(R).vec, [cos(pi/2) sin(pi/2)*[1, 0, 0]])
#         R = roty(-pi)
#         nt.assert_array_almost_equal( UnitQuaternion(R).vec, [cos(pi/2) sin(pi/2)*[0, 1, 0]])
#         R = rotz(-pi)
#         nt.assert_array_almost_equal( UnitQuaternion(R).vec, [cos(pi/2) sin(pi/2)*[0, 0, 1]])
    
#     def test_convert(self):
#         # test conversion from rotn matrix to u.quaternion and back
#         R = rotx(0)
#         nt.assert_array_almost_equal( UnitQuaternion(R).R, R)
        
#         R = rotx(pi/2)
#         nt.assert_array_almost_equal( UnitQuaternion(R).R, R)
#         R = roty(pi/2)
#         nt.assert_array_almost_equal( UnitQuaternion(R).R, R)
#         R = rotz(pi/2)
#         nt.assert_array_almost_equal( UnitQuaternion(R).R, R)
        
#         R = rotx(-pi/2)
#         nt.assert_array_almost_equal( UnitQuaternion(R).R, R)
#         R = roty(-pi/2)
#         nt.assert_array_almost_equal( UnitQuaternion(R).R, R)
#         R = rotz(-pi/2)
#         nt.assert_array_almost_equal( UnitQuaternion(R).R, R)
        
#         R = rotx(pi)
#         nt.assert_array_almost_equal( UnitQuaternion(R).R, R)
#         R = roty(pi)
#         nt.assert_array_almost_equal( UnitQuaternion(R).R, R)
#         R = rotz(pi)
#         nt.assert_array_almost_equal( UnitQuaternion(R).R, R)
        
#         R = rotx(-pi)
#         nt.assert_array_almost_equal( UnitQuaternion(R).R, R)
#         R = roty(-pi)
#         nt.assert_array_almost_equal( UnitQuaternion(R).R, R)
#         R = rotz(-pi)
#         nt.assert_array_almost_equal( UnitQuaternion(R).R, R)
    
    
#     def test_resulttype(self):
        
#         q = Quaternion([2, 0, 0, 0])
#         u = UnitQuaternion()
        
#         verifyClass(tc, q*q, 'Quaternion')
#         verifyClass(tc, q*u, 'Quaternion')
#         verifyClass(tc, u*q, 'Quaternion')
#         verifyClass(tc, u*u, 'UnitQuaternion')
        
#         verifyClass(tc, u.*u, 'UnitQuaternion')
#         # other combos all fail, test this?
        
#         verifyClass(tc, q/q, 'Quaternion')
#         verifyClass(tc, q/u, 'Quaternion')
#         verifyClass(tc, u/u, 'UnitQuaternion')
        
#         verifyClass(tc, conj(u), 'UnitQuaternion')
#         verifyClass(tc, inv(u), 'UnitQuaternion')
#         verifyClass(tc, unit(u), 'UnitQuaternion')
#         verifyClass(tc, unit(q), 'Quaternion')
        
#         verifyClass(tc, conj(q), 'Quaternion')
#         verifyClass(tc, inv(q), 'Quaternion')
        
#         verifyClass(tc, q+q, 'Quaternion')
#         verifyClass(tc, q-q, 'Quaternion')
        
#         verifyClass(tc, u.SO3, 'SO3')
#         verifyClass(tc, u.SE3, 'SE3')
    
    
#     def test_multiply(self):
        
#         vx = [1, 0, 0]'; vy = [0, 1, 0]'; vz = [0, 0, 1]'
#         rx = UnitQuaternion.Rx(pi/2)
#         ry = UnitQuaternion.Ry(pi/2)
#         rz = UnitQuaternion.Rz(pi/2)
#         u = UnitQuaternion()
        
#         # quat-quat product
#         # scalar x scalar
        
#         nt.assert_array_almost_equal(rx*u, rx)
#         nt.assert_array_almost_equal(u*rx, rx); 
        
#         #vector x vector
#         nt.assert_array_almost_equal([ry rz rx] * [rx ry rz], [ry*rx rz*ry rx*rz])
        
#         # scalar x vector
#         nt.assert_array_almost_equal(ry * [rx ry rz], [ry*rx ry*ry ry*rz])
        
#         #vector x scalar
#         nt.assert_array_almost_equal([rx ry rz] * ry, [rx*ry ry*ry rz*ry])
        
#         # quatvector product
#         # scalar x scalar
        
#         nt.assert_array_almost_equal(rx*vy, vz)
        
#         #vector x vector
#         nt.assert_array_almost_equal([ry rz rx] * [vz vx vy], [vx vy vz])
        
#         # scalar x vector
#         nt.assert_array_almost_equal(ry * [vx vy vz], [-vz vy vx])
        
#         #vector x scalar
#         nt.assert_array_almost_equal([ry rz rx] * vy, [vy -vx vz])
    
#     def multiply_test_normalized(self):
        
#         vx = [1, 0, 0]'; vy = [0, 1, 0]'; vz = [0, 0, 1]'
#         rx = UnitQuaternion.Rx(pi/2)
#         ry = UnitQuaternion.Ry(pi/2)
#         rz = UnitQuaternion.Rz(pi/2)
#         u = UnitQuaternion()
        
#         # quat-quat product
#         # scalar x scalar
        
#         nt.assert_array_almost_equal(double(rx.*u), double(rx))
#         nt.assert_array_almost_equal(double(u.*rx), double(rx))
        
#         # shouldn't make that much difference here
#         nt.assert_array_almost_equal(double(rx.*ry), double(rx*ry))
#         nt.assert_array_almost_equal(double(rx.*rz), double(rx*rz))
        
#         #vector x vector
#         nt.assert_array_almost_equal([ry rz rx] .* [rx ry rz], [ry.*rx rz.*ry rx.*rz])
        
#         # scalar x vector
#         nt.assert_array_almost_equal(ry .* [rx ry rz], [ry.*rx ry.*ry ry.*rz])
        
#         #vector x scalar
#         nt.assert_array_almost_equal([rx ry rz] .* ry, [rx.*ry ry.*ry rz.*ry])
    
#     def test_divide(self):
        
#         rx = UnitQuaternion.Rx(pi/2)
#         ry = UnitQuaternion.Ry(pi/2)
#         rz = UnitQuaternion.Rz(pi/2)
#         u = UnitQuaternion()
        
#         # scalar / scalar
#         # implicity tests inv
    
#         nt.assert_array_almost_equal(rx/u, rx)
#         nt.assert_array_almost_equal(ry/ry, u)
    
#         #vector /vector
#         nt.assert_array_almost_equal([ry rz rx] / [rx ry rz], [ry/rx rz/ry rx/rz])
        
#         #vector / scalar
#         nt.assert_array_almost_equal([rx ry rz] / ry, [rx/ry ry/ry rz/ry])
        
#         # scalar /vector
#         nt.assert_array_almost_equal(ry / [rx ry rz], [ry/rx ry/ry ry/rz])
    
    
#     def divide_test_normalized(self):
        
#         rx = UnitQuaternion.Rx(pi/2)
#         ry = UnitQuaternion.Ry(pi/2)
#         rz = UnitQuaternion.Rz(pi/2)
#         u = UnitQuaternion()
        
#         # scalar / scalar
        
#         # shouldn't make that much difference here
#         nt.assert_array_almost_equal(double(rx./ry), double(rx/ry))
#         nt.assert_array_almost_equal(double(rx./rz), double(rx/rz))
        
#         nt.assert_array_almost_equal(double(rx./u), double(rx))
#         nt.assert_array_almost_equal(double(ry./ry), double(u))
    
#         #vector /vector
#         nt.assert_array_almost_equal([ry rz rx] ./ [rx ry rz], [ry./rx rz./ry rx./rz])
        
#         #vector / scalar
#         nt.assert_array_almost_equal([rx ry rz] ./ ry, [rx./ry ry./ry rz./ry])
        
#        # scalar /vector
#         nt.assert_array_almost_equal(ry ./ [rx ry rz], [ry./rx ry./ry ry./rz])
    
    
#     def test_angle(self):
#             # angle between quaternions
#         # pure
#         v = [5, 6, 7]
    
    
#     def test_conversions(self):
        
#         #, 3 angle
#         nt.assert_array_almost_equal(UnitQuaternion.rpy(, 0.1, 0.2, 0.3 ).torpy, [, 0.1, 0.2, 0.3])
        
#         nt.assert_array_almost_equal(UnitQuaternion.eul(, 0.1, 0.2, 0.3 ).toeul, [, 0.1, 0.2, 0.3 ])
        
#         nt.assert_array_almost_equal(UnitQuaternion.rpy(, 10, 20, 30, 'deg' ).R, rpy2r(, 10, 20, 30, 'deg' ))
        
#         nt.assert_array_almost_equal(UnitQuaternion.eul(, 10, 20, 30, 'deg' ).R, eul2r(, 10, 20, 30, 'deg' ))
        
#         # (theta, v)
#         th =, 0.2; v = unit([1, 2, 3])
#         a = UnitQuaternion.an.vec(th, v ).toan.vec
#         nt.assert_array_almost_equal(a, th)
        
#         [a,b] = UnitQuaternion.an.vec(th, v ).toan.vec
#         nt.assert_array_almost_equal(a, th)
#         nt.assert_array_almost_equal(b, v)
        
#         # null rotation case
#         th =, 0; v = unit([1, 2, 3])
#         a = UnitQuaternion.an.vec(th, v ).toan.vec
#         nt.assert_array_almost_equal(a, th)
        
    
#     #  SO3                     convert to SO3 class
#     #  SE3                     convert to SE3 class
    
    
#     def test_miscellany(self):
        
#         # AbsTol not used since Quaternion supports eq() operator
        
#         rx = UnitQuaternion.Rx(pi/2)
#         ry = UnitQuaternion.Ry(pi/2)
#         rz = UnitQuaternion.Rz(pi/2)
#         u = UnitQuaternion()
        
#         # norm
#         nt.assert_array_almost_equal(rx.norm, 1)
#         nt.assert_array_almost_equal(norm([rx ry rz]), [1, 1, 1]')
        
#         # unit
#         nt.assert_array_almost_equal(rx.unit, rx)
#         nt.assert_array_almost_equal(unit([rx ry rz]), [rx ry rz])
        
#         # inner
#         nt.assert_array_almost_equal(u.inner(u), 1)
#         nt.assert_array_almost_equal(rx.inner(ry), 0.5)
#         nt.assert_array_almost_equal(rz.inner(rz), 1)
    
    
#         q = rx*ry*rz
            
#         nt.assert_array_almost_equal(q^0, u)
#         nt.assert_array_almost_equal(q^(-1), inv(q))
#         nt.assert_array_almost_equal(q^2, q*q)
        
#         # angle
#         nt.assert_array_almost_equal(angle(u, u), 0)
#         nt.assert_array_almost_equal(angle(u, rx), pi/4)
#         nt.assert_array_almost_equal(angle(u, [rx u]), pi/4*[1, 0])
#         nt.assert_array_almost_equal(angle([rx u], u), pi/4*[1, 0])
#         nt.assert_array_almost_equal(angle([rx u], [u rx]), pi/4*[1, 1])
    
        
#         # increment
#         w = [0.02, 0.03, 0.04]
        
#         nt.assert_array_almost_equal(rx.increment(w), rx*UnitQuaternion.omega(w))
    
#     def test_interp(self):
            
#         rx = UnitQuaternion.Rx(pi/2)
#         ry = UnitQuaternion.Ry(pi/2)
#         rz = UnitQuaternion.Rz(pi/2)
#         u = UnitQuaternion()
        
#         q = rx*ry*rz
        
#         # from null
#         nt.assert_array_almost_equal(q.interp(0), u)
#         nt.assert_array_almost_equal(q.interp(1), q )
        
#         nt.assert_array_almost_equal(length(q.interp(linspace(0,1, 10))), 10)
#         self.assertTrue(all( q.interp([0, 1]) == [u q]))
        
#         q0_5 = q.interp(0.5)
#         nt.assert_array_almost_equal( q0_5 * q0_5, q)
        
#         # between two quaternions
#         nt.assert_array_almost_equal(q.interp(rx, 0), q )
#         nt.assert_array_almost_equal(q.interp(rx, 1), rx )
        
#         self.assertTrue(all( q.interp(rx, [0, 1]) == [q rx]))
        
#         # test shortest option
#         q1 = UnitQuaternion.Rx(0.9*pi)
#         q2 = UnitQuaternion.Rx(-0.9*pi)
#         qq = q1.interp(q2, 11)
#         nt.assert_array_almost_equal( qq(6), UnitQuaternion.Rx(0) )
#         qq = q1.interp(q2, 11, 'shortest')
#         nt.assert_array_almost_equal( qq(6), UnitQuaternion.Rx(pi) )
    
    
#     def test_eq(self):
#         q1 = UnitQuaternion([0, 1, 0, 0])
#     	q2 = UnitQuaternion([0, -1, 0, 0])
#         q3 = UnitQuaternion.Rz(pi/2)
        
#         tc.verifyTrue( q1 == q1)
#         tc.verifyTrue( q2 == q2)
#         tc.verifyTrue( q3 == q3)
#         tc.verifyTrue( q1 == q2)
#         tc.verifyFalse( q1 == q3)
        
#         nt.assert_array_almost_equal( [q1 q1 q1] == [q1 q1 q1], [true true true])
#         nt.assert_array_almost_equal( [q1 q2 q3] == [q1 q2 q3], [true true true])
#         nt.assert_array_almost_equal( [q1 q1 q3] == q1, [true true false])
#         nt.assert_array_almost_equal( q3 == [q1 q1 q3], [false false true])
    
    
    
#     def test_logical(self):
#         rx = UnitQuaternion.Rx(pi/2)
#         ry = UnitQuaternion.Ry(pi/2)
        
#         # equality tests
#         self.assertTrue(rx == rx)
#         self.assertFalse(rx != rx)
#         self.assertFalse(rx == ry)
    
    
    
#     def test_dot(self):
#         q = UnitQuaternion()
#         omega = [1, 2, 3]
        
#         nt.assert_array_almost_equal(q.dot(omega).vec, [0 omega/2])
#         nt.assert_array_almost_equal(q.dotb(omega).vec, [0 omega/2])
        
#         q = UnitQuaternion.Rx(pi/2)
#         nt.assert_array_almost_equal(q.dot(omega), , 0.5*Quaternion.pure(omega)*q)
#         nt.assert_array_almost_equal(q.dotb(omega), 0.5*q*Quaternion.pure(omega))
    
#     def test_matrix(self):
        
#         q1 = UnitQuaternion.rpy(0.1, 0.2, 0.3)
#         q2 = UnitQuaternion.rpy(0.2, 0.3, 0.4)
        
#         q12 = q1 * q2
        
#         nt.assert_array_almost_equal(double(q12)', q1.matrix() * q2.vec')
    
    
#     def.test_vec3(self):
        
#         q1 = UnitQuaternion.rpy(0.1, 0.2, 0.3)
#         q2 = UnitQuaternion.rpy(0.2, 0.3, 0.4)
        
#         q12 = q1 * q2
        
#         q1v = q1.t.vec; q2v = q2.t.vec
        
#         q12v = UnitQuaternion.qvmul(q1v, q2v)
        
#         q12_ = UnitQuaternion.vec(q12v)
        
#         nt.assert_array_almost_equal(q12, q12_)
    
    
     
#     def test_display(self):
#             ry = UnitQuaternion.Ry(pi/2)
    
#             ry.plot()
#             h = ry.plot()
#             ry.animate()
#             ry.animate('rgb')
#             ry.animate( UnitQuaternion.Rx(pi/2), 'rgb' )

    pass
    
        
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