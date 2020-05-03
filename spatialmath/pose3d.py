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

class SO2(sp.SuperPose):
    
    # SO2()  identity matrix
    # SO2(angle, unit)
    # SO2( obj )   # deep copy
    # SO2( np )  # make numpy object
    # SO2( nplist )  # make from list of numpy objects
    
    # constructor needs to take ndarray -> SO2, or list of ndarray -> SO2
    def __init__(self, arg = None, *, unit='rad'):
        super().__init__()  # activate the UserList semantics
        
        if arg is None:
            # empty constructor
            self.data = [np.eye(2)]
        
        elif argcheck.isvector(arg):
            # SO2(value)
            # SO2(list of values)
            self.data = [tr.rot2(x, unit) for x in argcheck.getvector(arg)]
            
        else:
            super().arghandler(arg)

    @classmethod
    def rand(cls, *, range=[0, 2*math.pi], unit='rad', N=1):
        rand = np.random.uniform(low=range[0], high=range[1], size=N)  # random values in the range
        return cls([tr.rot2(x) for x in argcheck.getunit(rand, unit)])
    
    @classmethod
    def isvalid(self, x):
        return tr.isrot2(x, check=True)

    @property
    def T(self):
        return SO2(self.A.T)
    
    def inv(self):
        return SO2(self.A.T)
    
    # for symmetry with other 
    @classmethod
    def R(cls, theta, unit='rad'):
        return SO2([tr.rot1(x, unit) for x in argcheck.getvector(theta)])
    
    @property
    def angle(self):
        """Returns angle of SO2 object matrices in unit radians"""
        angles = []
        for each_matrix in self:
            angles.append(math.atan2(each_matrix[1, 0], each_matrix[0, 0]))
        # TODO !! Return list be default ?
        if len(angles) == 1:
            return angles[0]
        elif len(angles) > 1:
            return angles

class SE2(SO2):
    # constructor needs to take ndarray -> SO2, or list of ndarray -> SO2
    def __init__(self, x = None, y = None, theta = None, *, unit='rad'):
        super().__init__()  # activate the UserList semantics
        
        if x is None:
            # empty constructor
            self.data = [np.eye(3)]
        
        elif all(map(lambda x: isinstance(x, (int,float)), [x, y, theta])):
            # SE2(x, y, theta)
            self.data = [tr.trot2(theta, t=[x,y], unit=unit)]
            
        elif argcheck.isvector(x) and argcheck.isvector(y) and argcheck.isvector(theta):
            # SE2(xvec, yvec, tvec)
            xvec = argcheck.getvector(x)
            yvec = argcheck.getvector(y, dim=len(xvec))
            tvec = argcheck.getvector(theta, dim=len(xvec))
            self.data = [tr.trot2(_t, t=[_x, _y]) for (_x, _y, _t) in zip(xvec, yvec, argcheck.getunit(tvec, unit))]
            
        elif isinstance(x, np.ndarray) and y is None and theta is None:
            assert x.shape[1] == 3, 'array argument must be Nx3'
            self.data = [tr.trot2(_t, t=[_x, _y], unit=unit) for (_x, _y, _t) in x]
            
        else:
            super().arghandler(x)

    @property
    def T(self):
        raise NotImplemented('transpose is not meaningful for SE3 object')

    @classmethod
    def rand(cls, *, xrange=[-1, 1], yrange=[-1, 1], trange=[0, 2*math.pi], unit='rad', N=1):
        x = np.random.uniform(low=xrange[0], high=xrange[1], size=N)  # random values in the range
        y = np.random.uniform(low=yrange[0], high=yrange[1], size=N)  # random values in the range
        theta = np.random.uniform(low=trange[0], high=trange[1], size=N)  # random values in the range
        return cls([tr.trot2(t, t=[x,y]) for (t,x,y) in zip(x, y, argcheck.getunit(theta, unit))])
    
    @classmethod
    def isvalid(self, x):
        return tr.ishom2(x, check=True)

    @property
    def t(self):
        return self.A[:2,2]
    
    @property
    def R(self):
        return SO2(self.A[:2,:2])
    
    def inv(self):
        return SO2(self.A.T)
    ArithmeticError()
    
class SO3(sp.SuperPose):
    
    # SO2()  identity matrix
    # SO2(angle, unit)
    # SO2( obj )   # deep copy
    # SO2( np )  # make numpy object
    # SO2( nplist )  # make from list of numpy objects
    
    # constructor needs to take ndarray -> SO2, or list of ndarray -> SO2
    def __init__(self, arg = None, *, unit='rad', check=True):
        super().__init__()  # activate the UserList semantics
        
        if arg is None:
            # empty constructor
            self.data = [np.eye(3)]
        else:
            super().pose_arghandler(arg, check=check)

    @classmethod
    def rand(cls, *, range=[0, 2*math.pi], unit='rad', N=1):
        rand = np.random.uniform(low=range[0], high=range[1], size=N)  # random values in the range
        return cls([tr.rot2(x) for x in argcheck.getunit(rand, unit)])
    
    @classmethod
    def isvalid(self, x):
        return tr.isrot(x, check=True)

    @property
    def T(self):
        if len(self) == 1:
            return SO3(self.A.T)
        else:
            return SO3([x.T for x in self.A])
    
    def inv(self):
        if len(self) == 1:
            return SO3(self.A.T)
        else:
            return SO3([x.T for x in self.A])
    
    @property
    def n(self):
        return self.A[:,0]
       
    @property
    def o(self):
        return self.A[:,1]
        
    @property
    def a(self):
        return self.A[:,2]
    
    @classmethod
    def Rx(cls, theta, unit='rad'):
        return cls([tr.rotx(x, unit=unit) for x in argcheck.getvector(theta)], check=False)

    @classmethod
    def Ry(cls, theta, unit='rad'):
        return cls([tr.roty(x, unit=unit) for x in argcheck.getvector(theta)], check=False)

    @classmethod
    def Rz(cls, theta, unit='rad'):
        return cls([tr.rotz(x, unit=unit) for x in argcheck.getvector(theta)], check=False)

    @classmethod
    def rand(cls, N=1):
        return cls( [tr.q2r(tr.rand()) for i in range(0,N)], check=False)
        

    # 
    

    @classmethod
    def eul(cls, angles, unit='rad'):
        return cls(tr.eul2r(angles, unit=unit), check=False)

    @classmethod
    def rpy(cls, angles, order='zyx', unit='rad'):
        return cls(tr.rpy2r(angles, order=order, unit=unit), check=False)

    @classmethod
    def oa(cls, o, a):
        return cls(tr.oa2r(o, a), check=False)

    @classmethod
    def angvec(cls, theta, v, *, unit='rad'):
        return cls(tr.angvec2r(theta, v, unit=unit), check=False)

class SE3(sp.SuperPose):

    def __init__(self, arg = None, *, unit='rad', check=True):
        super().__init__()  # activate the UserList semantics
        
        if arg is None:
            # empty constructor
            self.data = [np.eye(4)]
        else:
            super().pose_arghandler(arg, check=check)

    @property
    def T(self):
        raise NotImplemented('transpose is not meaningful for SE3 object')
    
    @classmethod
    def rand(cls, *, range=[0, 2*math.pi], unit='rad', N=1):
        rand = np.random.uniform(low=range[0], high=range[1], size=N)  # random values in the range
        return cls([tr.rot2(x) for x in argcheck.getunit(rand, unit)])
    
    @classmethod
    def isvalid(self, x):
        return tr.ishom(x, check=True)
    
    @classmethod
    def Rx(cls, theta, unit='rad'):
        return cls([tr.trotx(x, unit) for x in argcheck.getvector(theta)])

    @classmethod
    def Ry(cls, theta, unit='rad'):
        return cls([tr.troty(x, unit) for x in argcheck.getvector(theta)])

    @classmethod
    def Rz(cls, theta, unit='rad'):
        return cls([tr.trotz(x, unit) for x in argcheck.getvector(theta)])
    
    @classmethod
    def Tx(cls, x):
        return cls(tr.transl(x, 0, 0))

    @classmethod
    def Ty(cls, y):
        return cls(tr.transl(0, y, 0))

    @classmethod
    def Tz(cls, z):
        return cls(tr.transl(0, 0, z))
    
    @classmethod
    def trans(cls, x = None, y = None, z = None):
        return cls(tr.transl(x, y, z))
    
    def inv(self):
        if len(self) == 1:
            return SO3(self.A.T)
        else:
            return SO3([x.T for x in self.A])
    
    @property
    def n(self):
        return self.A[:3,0]
       
    @property
    def o(self):
        return self.A[:3,1]
        
    @property
    def a(self):
        return self.A[:3,2]
    
    @property
    def t(self):
        return self.A[:3,3]
    
    

    @classmethod
    def rand(cls, *, xrange=[-1, 1], yrange=[-1, 1], zrange=[-1, 1], N=1):
        X = np.random.uniform(low=xrange[0], high=xrange[1], size=N)  # random values in the range
        Y = np.random.uniform(low=yrange[0], high=yrange[1], size=N)  # random values in the range
        Z = np.random.uniform(low=yrange[0], high=zrange[1], size=N)  # random values in the range
        R = SO3.rand(N=N)
        return cls([tr.transl(x, y, z) @ tr.r2t(r.A) for (x,y,z,r) in zip(X, Y, Z, R)])

    @classmethod
    def eul(cls, angles, unit='rad'):
        return cls(tr.eul2tr(angles, unit=unit))

    @classmethod
    def rpy(cls, angles, order='zyx', unit='rad'):
        return cls(tr.rpy2tr(angles, order=order, unit=unit))

    @classmethod
    def oa(cls, o, a):
        return cls(tr.oa2tr(o, a))

    @classmethod
    def angvec(cls, theta, v, *, unit='rad'):
        return cls(tr.angvec2tr(theta, v, unit=unit))


if __name__ == '__main__':
    
    R = SO3()

    # print(isinstance(R, SO3))
    # print(isinstance(R, sp.SuperPose))

    # a = SO2(0.2)
    # b = SO2(a)
    # print(a+a)
    # print(a*a)

    # b = SO2(0.1)
    # b.append(a)
    # b.append(a)
    # b.append(a)
    # b.append(a)
    # print(len(a))
    # print(len(b))
    # print(b)

    # c = SO2(0.3)
    # c.extend(a)
    # c.extend(b)
    # print(len(c))

    # d = SO2(0.4)
    # d.append(b)
    # print(len(d))
    # print(d)


    # if __name__ == '__main__':

    #     import numpy.testing as nt
            
    #     class Test_check(unittest.TestCase):
            
    #         def test_unit(self):
                

    #print(a)
    #print(a*a)
    #c = SO2(0)
    #
    #b = a
    #print(len(b))
    #b.append(c)
    #b.append(c)
    #print(len(b))
    #print(b)
    #print(b)
    #print(b[0])
    #print(type(a))
    #print(type(b))

    #arr = [SO2(0), SO2(0.1), SO2(0.2)]
    #print(arr)
    #b = np.array(arr)
    #print(b)
    #print('--')
    #print(arr[0])
    #print(b[1])
    #print(b*a)
    

    # import pathlib
    # import os.path
    
    # runfile(os.path.join(pathlib.Path(__file__).parent.absolute(), "test_tr.py") )
 
    
    import pathlib
    import os.path
    
    runfile(os.path.join(pathlib.Path(__file__).parent.absolute(), "test_pose.py") )
    
    T = SE3()