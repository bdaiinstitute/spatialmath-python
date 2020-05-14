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
    def __init__(self, arg=None, *, unit='rad'):
        super().__init__()  # activate the UserList semantics

        if arg is None:
            # empty constructor
            if type(self) is SO2:
                self.data = [np.eye(2)]

        elif argcheck.isvector(arg):
            # SO2(value)
            # SO2(list of values)
            self.data = [tr.rot2(x, unit) for x in argcheck.getvector(arg)]

        elif isinstance(arg, np.ndarray) and arg.shape == (2,2):
            self.data = [arg]
        else:
            super().arghandler(arg)

    @classmethod
    def rand(cls, *, range=[0, 2 * math.pi], unit='rad', N=1):
        rand = np.random.uniform(low=range[0], high=range[1], size=N)  # random values in the range
        return cls([tr.rot2(x) for x in argcheck.getunit(rand, unit)])

    @staticmethod
    def isvalid(x):
        return tr.isrot2(x, check=True)

    @property
    def T(self):
        return SO2(self.A.T)

    @property
    def inv(self):
        if len(self) == 1:
            return SO2(self.A.T)
        else:
            return SO2([x.T for x in self.A])

    @property
    def R(self):
        return self.A[:2, :2]

    @property
    def theta(self):
        """Returns angle of SO2 object matrices in unit radians"""
        if len(self) == 1:
            return math.atan2(self.A[1,0], self.A[0,0])
        else:
            return [math.atan2(x.A[1,0], x.A[0,0]) for x in self]        


class SE2(SO2):
    # constructor needs to take ndarray -> SO2, or list of ndarray -> SO2
    def __init__(self, x=None, y=None, theta=None, *, unit='rad'):
        super().__init__()  # activate the UserList semantics

        if x is None and y is None and theta is None:
            # SE2()
            # empty constructor
            self.data = [np.eye(3)]

        elif x is not None:
            if y is not None and theta is not None:
                # SE2(x, y, theta)
                self.data = [tr.trot2(theta, t=[x, y], unit=unit)]

            elif y is None and theta is None:
                if argcheck.isvector(x, 3):
                    # SE2( [x,y,theta])
                    self.data = [tr.trot2(x[2], t=x[:2], unit=unit)]
                elif isinstance(x, np.ndarray):
                    if x.shape == (3,3):
                        # SE2( 3x3 matrix )
                        self.data = [x]
                    elif x.shape[1] == 3:
                        # SE2( Nx3 )
                        self.data = [tr.trot2(T.theta, t=T.t) for T in x]
                else:
                    super().arghandler(x)
        else:
            raise ValueError('bad arguments to constructor')

    @property
    def T(self):
        raise NotImplemented('transpose is not meaningful for SE3 object')

    @classmethod
    def rand(cls, *, xrange=[-1, 1], yrange=[-1, 1], trange=[0, 2 * math.pi], unit='rad', N=1):
        x = np.random.uniform(low=xrange[0], high=xrange[1], size=N)  # random values in the range
        y = np.random.uniform(low=yrange[0], high=yrange[1], size=N)  # random values in the range
        theta = np.random.uniform(low=trange[0], high=trange[1], size=N)  # random values in the range
        return cls([tr.trot2(t, t=[x, y]) for (t, x, y) in zip(x, y, argcheck.getunit(theta, unit))])

    @staticmethod
    def isvalid(x):
        return tr.ishom2(x, check=True)

    @property
    def t(self):
        return self.A[:2, 2]

    @property
    def xyt(self):
        if len(self) == 1:
            return np.r_[self.t, self.theta]
        else:
            return [np.r_[x.t, x.theta] for x in self]

    @property
    def inv(self):
        if len(self) == 1:
            return SE2(tr.rt2tr(self.R.T, -self.R.T @ self.t))
        else:
            return SE2([tr.rt2tr(x.R.T, -x.R.T @ x.t) for x in self])


if __name__ == '__main__':  # pragma: no cover

    import pathlib
    import os.path

    exec(open(os.path.join(pathlib.Path(__file__).parent.absolute(), "test_pose2d.py")).read())
