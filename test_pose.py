import numpy.testing as nt
import unittest

"""
we will assume that the primitives rotx,trotx, etc. all work
"""
from math import pi
from spatialmath.pose import *
from spatialmath import super_pose as sp
from spatialmath.base import *
import spatialmath.base.argcheck as argcheck

def array_compare(x, y):
    if isinstance(x, sp.SuperPose):
        x = x.A
    if isinstance(y, sp.SuperPose):
        y = y.A
    nt.assert_array_almost_equal(x, y)
                           
                           
class TestSO2(unittest.TestCase):
    pass

class TestSE2(unittest.TestCase):
    pass


    

class TestSO3(unittest.TestCase):
    
    def test_constructor(self):
        
        # null constructor
        R = SO3()
        nt.assert_equal(len(R),  1)
        array_compare(R, np.eye(3))
        nt.assert_equal(isinstance(R, SO3), True)
        
        # construct from matrix
        R = SO3( rotx(0.2) )
        nt.assert_equal(len(R),  1)
        array_compare(R, rotx(0.2))
        nt.assert_equal(isinstance(R, SO3), True)
        
        # construct from canonic rotation
        R = SO3.Rx(0.2)
        nt.assert_equal(len(R),  1)
        array_compare(R, rotx(0.2))
        nt.assert_equal(isinstance(R, SO3), True)
        
        R = SO3.Ry(0.2)
        nt.assert_equal(len(R),  1)
        array_compare(R, roty(0.2))
        nt.assert_equal(isinstance(R, SO3), True)
        
        R = SO3.Rz(0.2)
        nt.assert_equal(len(R),  1)
        array_compare(R, rotz(0.2))
        nt.assert_equal(isinstance(R, SO3), True)
        
        # triple angle
        R = SO3.eul([0.1, 0.2, 0.3])
        nt.assert_equal(len(R),  1)
        array_compare(R, eul2r([0.1, 0.2, 0.3]))
        nt.assert_equal(isinstance(R, SO3), True)
        
        R = SO3.eul(np.r_[0.1, 0.2, 0.3])
        nt.assert_equal(len(R),  1)
        array_compare(R, eul2r([0.1, 0.2, 0.3]))
        nt.assert_equal(isinstance(R, SO3), True)
        
        R = SO3.eul([10, 20, 30], unit='deg')
        nt.assert_equal(len(R),  1)
        array_compare(R, eul2r([10, 20, 30], unit='deg'))
        nt.assert_equal(isinstance(R, SO3), True)
        
        R = SO3.rpy([0.1, 0.2, 0.3])
        nt.assert_equal(len(R),  1)
        array_compare(R, rpy2r([0.1, 0.2, 0.3]))
        nt.assert_equal(isinstance(R, SO3), True)
        
        R = SO3.rpy(np.r_[0.1, 0.2, 0.3])
        nt.assert_equal(len(R),  1)
        array_compare(R, rpy2r([0.1, 0.2, 0.3]))
        nt.assert_equal(isinstance(R, SO3), True)
        
        R = SO3.rpy([10, 20, 30], unit='deg')
        nt.assert_equal(len(R),  1)
        array_compare(R, rpy2r([10, 20, 30], unit='deg'))
        nt.assert_equal(isinstance(R, SO3), True)
        
        R = SO3.rpy([0.1, 0.2, 0.3], order='xyz')
        nt.assert_equal(len(R),  1)
        array_compare(R, rpy2r([0.1, 0.2, 0.3], order='xyz'))
        nt.assert_equal(isinstance(R, SO3), True)
             
        # angvec
        R = SO3.angvec(0.2, [1, 0, 0])
        nt.assert_equal(len(R),  1)
        array_compare(R, rotx(0.2))
        nt.assert_equal(isinstance(R, SO3), True)
        
        R = SO3.angvec(0.3, [0, 1, 0])
        nt.assert_equal(len(R),  1)
        array_compare(R, roty(0.3))
        nt.assert_equal(isinstance(R, SO3), True)
        
        # OA
        R = SO3.oa([0, 1, 0], [0, 0, 1])
        nt.assert_equal(len(R),  1)
        array_compare(R, np.eye(3))
        nt.assert_equal(isinstance(R, SO3), True)
        
        # random
        R = SO3.rand()
        nt.assert_equal(len(R),  1)
        nt.assert_equal(isinstance(R, SO3), True)
        
        # copy constructor
        R = SO3.Rx(pi/2)
        R2 = SO3(R)
        R = SO3.Ry(pi/2)
        array_compare(R2, rotx(pi/2))
        
        
    def test_listpowers(self):
        R = SO3()
        R1 = SO3.Rx(0.2)
        R2 = SO3.Ry(0.3)
        
        R.append(R1)
        R.append(R2)
        nt.assert_equal(len(R),  3)
        nt.assert_equal(isinstance(R, SO3), True)
        
        array_compare(R[0], np.eye(3))
        array_compare(R[1], R1)
        array_compare(R[2], R2)
        
        R = SO3([rotx(0.1), rotx(0.2), rotx(0.3)])
        nt.assert_equal(len(R),  3)
        nt.assert_equal(isinstance(R, SO3), True)
        array_compare(R[0], rotx(0.1))
        array_compare(R[1], rotx(0.2))
        array_compare(R[2], rotx(0.3))
        
        R = SO3([SO3.Rx(0.1), SO3.Rx(0.2), SO3.Rx(0.3)])
        nt.assert_equal(len(R),  3)
        nt.assert_equal(isinstance(R, SO3), True)
        array_compare(R[0], rotx(0.1))
        array_compare(R[1], rotx(0.2))
        array_compare(R[2], rotx(0.3))
        
    def test_tests(self):
        
        R = SO3()
        
        self.assertEqual( R.isrot(), True)
        self.assertEqual( R.isrot2(), False)
        self.assertEqual( R.ishom(), False)
        self.assertEqual( R.ishom2(), False)
           
    
        
    def test_properties(self):

        R = SO3()
        
        self.assertEqual( R.isSO, True)
        self.assertEqual( R.isSE, False)
        
        array_compare(R.n, np.r_[1, 0, 0])
        array_compare(R.n, np.r_[1, 0, 0])
        array_compare(R.n, np.r_[1, 0, 0])
        
        nt.assert_equal(R.N, 3)
        nt.assert_equal(R.shape, (3,3))
        
        R = SO3.Rx(0.3)
        array_compare(R.T * R, np.eye(3,3))
        
    def test_arith(self):
        R = SO3()
        
        # sum
        a = R + R
        nt.assert_equal(isinstance(a, SO3), False)
        array_compare(a, np.array([ [2,0,0], [0,2,0], [0,0,2]]))
        
        a = R + 1
        nt.assert_equal(isinstance(a, SO3), False)
        array_compare(a, np.array([ [2,1,1], [1,2,1], [1,1,2]]))
        
        # a = 1 + R
        # nt.assert_equal(isinstance(a, SO3), False)
        # array_compare(a, np.array([ [2,1,1], [1,2,1], [1,1,2]]))
        
        a = R + np.eye(3)
        nt.assert_equal(isinstance(a, SO3), False)
        array_compare(a, np.array([ [2,0,0], [0,2,0], [0,0,2]]))
        
        # a =  np.eye(3) + R
        # nt.assert_equal(isinstance(a, SO3), False)
        # array_compare(a, np.array([ [2,0,0], [0,2,0], [0,0,2]]))
        #  this invokes the __add__ method for numpy
        
        R += R
        nt.assert_equal(isinstance(a, SO3), False)
        array_compare(a, np.array([ [2,0,0], [0,2,0], [0,0,2]]))
        
        R = SO3()
        R += 1
        nt.assert_equal(isinstance(R, SO3), False)
        array_compare(R, np.array([ [2,1,1], [1,2,1], [1,1,2]]))
        
        # difference
        R = SO3()
         
        a = R - R
        nt.assert_equal(isinstance(a, SO3), False)
        array_compare(a, np.zeros((3,3)))
        
        a = R - 1
        nt.assert_equal(isinstance(a, SO3), False)
        array_compare(a, np.array([ [0,-1,-1], [-1,0,-1], [-1,-1,0]]))
        
        # a = 1 - R
        # nt.assert_equal(isinstance(a, SO3), False)
        # array_compare(a, -np.array([ [0,-1,-1], [-1,0,-1], [-1,-1,0]]))
        
        a = R - np.eye(3)
        nt.assert_equal(isinstance(a, SO3), False)
        array_compare(a, np.zeros((3,3)))

        # a =  np.eye(3) - R
        # nt.assert_equal(isinstance(a, SO3), False)
        # array_compare(a, np.zeros((3,3)))
        
        R -= R
        nt.assert_equal(isinstance(R, SO3), False)
        array_compare(R, np.zeros((3,3)))
        
        R = SO3()
        R -= 1
        nt.assert_equal(isinstance(R, SO3), False)
        array_compare(R, np.array([ [0,-1,-1], [-1,0,-1], [-1,-1,0]]))

        # multiply
        R = SO3()
        
        a = R * R
        nt.assert_equal(isinstance(a, SO3), True)
        array_compare(a, R)
        
        a = R * 2
        nt.assert_equal(isinstance(a, SO3), False)
        array_compare(a, 2*np.eye(3))
        
        a = 2 * R
        nt.assert_equal(isinstance(a, SO3), False)
        array_compare(a, 2*np.eye(3))
        
        R = SO3()
        R *= SO3.Rx(pi/2)
        nt.assert_equal(isinstance(R, SO3), True)
        array_compare(R, rotx(pi/2))
        
        R = SO3()
        R *= 2
        nt.assert_equal(isinstance(R, SO3), False)
        array_compare(R, 2*np.eye(3))
        
        array_compare(SO3.Rx(pi/2) * SO3.Ry(pi/2) * SO3.Rx(-pi/2), SO3.Rz(pi/2))
    
        array_compare(SO3.Ry(pi/2) * [1, 0, 0], np.c_[0,0,-1].T)
        
        # SO3 x vector
        vx = np.r_[1, 0, 0]
        vy = np.r_[0, 1, 0]
        vz = np.r_[0, 0, 1]
        
        def cv(v):
            return np.c_[v]
        
        nt.assert_equal(isinstance(SO3.Rx(pi/2) * vx, np.ndarray), True)
        array_compare(SO3.Rx(pi/2) * vx, cv(vx))
        array_compare(SO3.Rx(pi/2) * vy, cv(vz))
        array_compare(SO3.Rx(pi/2) * vz, cv(-vy))
        
        array_compare(SO3.Ry(pi/2) * vx, cv(-vz))
        array_compare(SO3.Ry(pi/2) * vy, cv(vy))
        array_compare(SO3.Ry(pi/2) * vz, cv(vx))
        
        array_compare(SO3.Rz(pi/2) * vx, cv(vy))
        array_compare(SO3.Rz(pi/2) * vy, cv(-vx))
        array_compare(SO3.Rz(pi/2) * vz, cv(vz))

        # divide
        R = SO3.Ry(0.3)
        a = R / R
        nt.assert_equal(isinstance(a, SO3), True)
        array_compare(a, np.eye(3))
        
        a = R / 2
        nt.assert_equal(isinstance(a, SO3), False)
        array_compare(a, roty(0.3)/2)


        
    def test_arith_vect(self):

        rx = SO3.Rx(pi/2)
        ry = SO3.Ry(pi/2)
        rz = SO3.Rz(pi/2)
        u = SO3()
        
        # multiply
        R = SO3([rx, ry, rz])
        a = R * rx
        nt.assert_equal(isinstance(a, SO3), True)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx*rx)
        array_compare(a[1], ry*rx)
        array_compare(a[2], rz*rx)
        
        a = rx * R
        nt.assert_equal(isinstance(a, SO3), True)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx*rx)
        array_compare(a[1], rx*ry)
        array_compare(a[2], rx*rz)
        
        a = R * R
        nt.assert_equal(isinstance(a, SO3), True)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx*rx)
        array_compare(a[1], ry*ry)
        array_compare(a[2], rz*rz)
                       
        a = R * 2
        nt.assert_equal(isinstance(a, SO3), False)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx*2)
        array_compare(a[1], ry*2)
        array_compare(a[2], rz*2)
        
        a = 2 * R
        nt.assert_equal(isinstance(a, SO3), False)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx*2)
        array_compare(a[1], ry*2)
        array_compare(a[2], rz*2)
        
        a = R
        a *= rx
        nt.assert_equal(isinstance(a, SO3), True)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx*rx)
        array_compare(a[1], ry*rx)
        array_compare(a[2], rz*rx)
        
        a = rx
        a *= R
        nt.assert_equal(isinstance(a, SO3), True)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx*rx)
        array_compare(a[1], rx*ry)
        array_compare(a[2], rx*rz)        
        
        a = R
        a *= R
        nt.assert_equal(isinstance(a, SO3), True)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx*rx)
        array_compare(a[1], ry*ry)
        array_compare(a[2], rz*rz)  
        
        a = R
        a *= 2
        nt.assert_equal(isinstance(a, SO3), False)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx*2)
        array_compare(a[1], ry*2)
        array_compare(a[2], rz*2)
        
        # SO3 x vector
        vx = np.r_[1, 0, 0]
        vy = np.r_[0, 1, 0]
        vz = np.r_[0, 0, 1]
        
        a = R * vx
        array_compare(a[:,0], (rx*vx).flatten())
        array_compare(a[:,1], (ry*vx).flatten())
        array_compare(a[:,2], (rz*vx).flatten())
        
        a = rx * np.vstack((vx,vy,vz)).T
        array_compare(a[:,0], (rx*vx).flatten())
        array_compare(a[:,1], (rx*vy).flatten())
        array_compare(a[:,2], (rx*vz).flatten())
        
        
        # divide
        R = SO3([rx, ry, rz])
        a = R / rx
        nt.assert_equal(isinstance(a, SO3), True)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx/rx)
        array_compare(a[1], ry/rx)
        array_compare(a[2], rz/rx)
        
        a = rx / R
        nt.assert_equal(isinstance(a, SO3), True)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx/rx)
        array_compare(a[1], rx/ry)
        array_compare(a[2], rx/rz)
        
        a = R / R
        nt.assert_equal(isinstance(a, SO3), True)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], np.eye(3))
        array_compare(a[1], np.eye(3))
        array_compare(a[2], np.eye(3))
        
        a = R / 2
        nt.assert_equal(isinstance(a, SO3), False)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx/2)
        array_compare(a[1], ry/2)
        array_compare(a[2], rz/2)
        
        a = R
        a /= rx
        nt.assert_equal(isinstance(a, SO3), True)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx/rx)
        array_compare(a[1], ry/rx)
        array_compare(a[2], rz/rx)
        
        a = rx
        a /= R
        nt.assert_equal(isinstance(a, SO3), True)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx/rx)
        array_compare(a[1], rx/ry)
        array_compare(a[2], rx/rz)
        
        a = R
        a /= R
        nt.assert_equal(isinstance(a, SO3), True)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], np.eye(3))
        array_compare(a[1], np.eye(3))
        array_compare(a[2], np.eye(3))
        
        a = R
        a /= 2
        nt.assert_equal(isinstance(a, SO3), False)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx/2)
        array_compare(a[1], ry/2)
        array_compare(a[2], rz/2)
        
        # add
        R = SO3([rx, ry, rz])
        a = R + rx
        nt.assert_equal(isinstance(a, SO3), False)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx+rx)
        array_compare(a[1], ry+rx)
        array_compare(a[2], rz+rx)
        
        a = rx + R
        nt.assert_equal(isinstance(a, SO3), False)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx+rx)
        array_compare(a[1], rx+ry)
        array_compare(a[2], rx+rz)
        
        a = R + R
        nt.assert_equal(isinstance(a, SO3), False)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx+rx)
        array_compare(a[1], ry+ry)
        array_compare(a[2], rz+rz)
        
        a = R + 1
        nt.assert_equal(isinstance(a, SO3), False)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx+1)
        array_compare(a[1], ry+1)
        array_compare(a[2], rz+1)
        
        a = R
        a += rx
        nt.assert_equal(isinstance(a, SO3), False)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx+rx)
        array_compare(a[1], ry+rx)
        array_compare(a[2], rz+rx)
        
        a = rx
        a += R
        nt.assert_equal(isinstance(a, SO3), False)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx+rx)
        array_compare(a[1], rx+ry)
        array_compare(a[2], rx+rz)
        
        a = R
        a += R
        nt.assert_equal(isinstance(a, SO3), False)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx+rx)
        array_compare(a[1], ry+ry)
        array_compare(a[2], rz+rz)
        
        a = R
        a += 1
        nt.assert_equal(isinstance(a, SO3), False)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx+1)
        array_compare(a[1], ry+1)
        array_compare(a[2], rz+1)        
        
        # subtract
        R = SO3([rx, ry, rz])
        a = R - rx
        nt.assert_equal(isinstance(a, SO3), False)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx-rx)
        array_compare(a[1], ry-rx)
        array_compare(a[2], rz-rx)
        
        a = rx - R
        nt.assert_equal(isinstance(a, SO3), False)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx-rx)
        array_compare(a[1], rx-ry)
        array_compare(a[2], rx-rz)
        
        a = R - R
        nt.assert_equal(isinstance(a, SO3), False)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx-rx)
        array_compare(a[1], ry-ry)
        array_compare(a[2], rz-rz)
        
        a = R - 1
        nt.assert_equal(isinstance(a, SO3), False)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx-1)
        array_compare(a[1], ry-1)
        array_compare(a[2], rz-1)
        
        a = R
        a -= rx
        nt.assert_equal(isinstance(a, SO3), False)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx-rx)
        array_compare(a[1], ry-rx)
        array_compare(a[2], rz-rx)
        
        a = rx
        a -= R
        nt.assert_equal(isinstance(a, SO3), False)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx-rx)
        array_compare(a[1], rx-ry)
        array_compare(a[2], rx-rz)
        
        a = R
        a -= R
        nt.assert_equal(isinstance(a, SO3), False)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx-rx)
        array_compare(a[1], ry-ry)
        array_compare(a[2], rz-rz)
        
        a = R
        a -= 1
        nt.assert_equal(isinstance(a, SO3), False)
        nt.assert_equal(len(a), 3)
        array_compare(a[0], rx-1)
        array_compare(a[1], ry-1)
        array_compare(a[2], rz-1)

        
    def test_functions(self):
        # inv
        # .T
        pass
    
    def test_functions_vect(self):
        # inv
        # .T
        pass

                
class TestSE3(unittest.TestCase):
    
    def test_constructor(self):
        #array_compare(UnitQuaternion().vec, np.r_[1,0,0,0])
        pass
        
# ---------------------------------------------------------------------------------------#
if __name__ == '__main__':

    
    unittest.main()
        
