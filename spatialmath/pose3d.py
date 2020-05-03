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


    
class SO3(sp.SMPose):
    
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
    def Rand(cls, N=1):
        return cls( [tr.q2r(tr.rand()) for i in range(0,N)], check=False)
        

    # 
    

    @classmethod
    def Eul(cls, angles, unit='rad'):
        return cls(tr.eul2r(angles, unit=unit), check=False)

    @classmethod
    def RPY(cls, angles, order='zyx', unit='rad'):
        return cls(tr.rpy2r(angles, order=order, unit=unit), check=False)

    @classmethod
    def OA(cls, o, a):
        return cls(tr.oa2r(o, a), check=False)

    @classmethod
    def AngVec(cls, theta, v, *, unit='rad'):
        return cls(tr.angvec2r(theta, v, unit=unit), check=False)

class SE3(sp.SMPose):

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
            return SE3(tr.rt2tr(self.R.T, -self.R.T @ self.t))
        else:
            return SE3([SE3(tr.rt2tr(x.R.T, -x.R.T @ x.t)) for x in self])
    
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
    
    @property
    def R(self):
        return self.A[:3,:3]
    
    @property
    def eul(self, **kwargs):
        return tr.tr2eul(self.A)
    
    @property
    def rpy(self, **kwargs):
        return tr.tr2eul(self.A, **kwargs)
    

    @classmethod
    def Rand(cls, *, xrange=[-1, 1], yrange=[-1, 1], zrange=[-1, 1], N=1):
        X = np.random.uniform(low=xrange[0], high=xrange[1], size=N)  # random values in the range
        Y = np.random.uniform(low=yrange[0], high=yrange[1], size=N)  # random values in the range
        Z = np.random.uniform(low=yrange[0], high=zrange[1], size=N)  # random values in the range
        R = SO3.Rand(N=N)
        return cls([tr.transl(x, y, z) @ tr.r2t(r.A) for (x,y,z,r) in zip(X, Y, Z, R)])

    @classmethod
    def Eul(cls, angles, unit='rad'):
        return cls(tr.eul2tr(angles, unit=unit))

    @classmethod
    def RPY(cls, angles, order='zyx', unit='rad'):
        return cls(tr.rpy2tr(angles, order=order, unit=unit))

    @classmethod
    def OA(cls, o, a):
        return cls(tr.oa2tr(o, a))

    @classmethod
    def AngVec(cls, theta, v, *, unit='rad'):
        return cls(tr.angvec2tr(theta, v, unit=unit))


if __name__ == '__main__':
    
    import pathlib
    import os.path
    
    runfile(os.path.join(pathlib.Path(__file__).parent.absolute(), "test_pose3d.py") )
