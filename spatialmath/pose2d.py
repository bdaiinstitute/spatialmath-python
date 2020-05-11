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

class SO2(sp.SMPose):
    
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

class SE2(sp.SMPose):
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
    

if __name__ == '__main__':  # pragma: no cover
 
    import pathlib
    import os.path
    
    exec(open(os.path.join(pathlib.Path(__file__).parent.absolute(), "test_pose2d.py")).read() )
    