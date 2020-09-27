#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 14:22:36 2020

@author: corkep
"""


if __name__ == '__main__':

    import timeit

    N = 100000

    transforms_setup = '''
import spatialmath as sm
import numpy as np
from collections import namedtuple
Rt = namedtuple('Rt', 'R t')
X1 = sm.SE3.Rand()
X2 = sm.SE3.Rand()
T1 = X1.A
T2 = X2.A
R1 = sm.base.t2r(T1)
R2 = sm.base.t2r(T2)
t1 = sm.base.transl(T1)
t2 = sm.base.transl(T2)
Rt1 = Rt(R1, t1)
Rt2 = Rt(R2, t2)
'''
    t = timeit.timeit(stmt='sm.base.getvector(0.2)', setup=transforms_setup, number=N)
    print(f"getvector(x):        {t/N*1e6:.3g} μs")

    t = timeit.timeit(stmt='sm.base.rotx(0.2, unit="rad")', setup=transforms_setup, number=N)
    print(f"transforms.rotx:     {t/N*1e6:.3g} μs")

    t = timeit.timeit(stmt='sm.base.trotx(0.2, unit="rad")', setup=transforms_setup, number=N)
    print(f"transforms.trotx:    {t/N*1e6:.3g} μs")

    t = timeit.timeit(stmt='sm.base.t2r(T1)', setup=transforms_setup, number=N)
    print(f"transforms.t2r:      {t/N*1e6:.3g} μs")

    t = timeit.timeit(stmt='sm.base.r2t(R1)', setup=transforms_setup, number=N)
    print(f"transforms.r2t:      {t/N*1e6:.3g} μs")

    t = timeit.timeit(stmt='sm.SE3.Rx(0.2)', setup=transforms_setup, number=N)
    print(f"SE3.Rx:              {t/N*1e6:.3g} μs")

    t = timeit.timeit(stmt='T1[:3,:3]', setup=transforms_setup, number=N)
    print(f"T1[:3,:3]:           {t/N*1e6:.3g} μs")

    t = timeit.timeit(stmt='X1.A', setup=transforms_setup, number=N)
    print(f"SE3.A:               {t/N*1e6:.3g} μs")

    t = timeit.timeit(stmt='sm.SE3()', setup=transforms_setup, number=N)
    print(f"SE3():               {t/N*1e6:.3g} μs")

    t = timeit.timeit(stmt='sm.SE3(T1)', setup=transforms_setup, number=N)
    print(f"SE3(T1):             {t/N*1e6:.3g} μs")

    t = timeit.timeit(stmt='sm.SE3(T1, check=False)', setup=transforms_setup, number=N)
    print(f"SE3(T1 check=False): {t/N*1e6:.3g} μs")

    t = timeit.timeit(stmt='sm.SE3([T1], check=False)', setup=transforms_setup, number=N)
    print(f"SE3([T1]):           {t/N*1e6:.3g} μs")

    t = timeit.timeit(stmt='X1 * X2', setup=transforms_setup, number=N)
    print(f"SE3 *:               {t/N*1e6:.3g} μs")

    t = timeit.timeit(stmt='T1 @ T2', setup=transforms_setup, number=N)
    print(f"4x4 @:               {t/N*1e6:.3g} μs")

    t = timeit.timeit(stmt='T1[:3,:3] @ T2[:3,:3] + T1[:3,:3] @ T2[:3,3]', setup=transforms_setup, number=N)
    print(f"R1*R2, R1*t:         {t/N*1e6:.3g} μs")

    t = timeit.timeit(stmt='(Rt1.R @ Rt2.R, Rt1.R @ Rt2.t)', setup=transforms_setup, number=N)
    print(f"T1 * T2 (R, t):      {t/N*1e6:.3g} μs")

    t = timeit.timeit(stmt='X1.inv()', setup=transforms_setup, number=N)
    print(f"SE3.inv:             {t/N*1e6:.3g} μs")

    t = timeit.timeit(stmt='sm.base.trinv(T1)', setup=transforms_setup, number=N)
    print(f"base.trinv:          {t/N*1e6:.3g} μs")

    t = timeit.timeit(stmt='(Rt1.R.T, -Rt1.R.T @ Rt1.t)', setup=transforms_setup, number=N)
    print(f"base.trinv (R,t):    {t/N*1e6:.3g} μs")

    t = timeit.timeit(stmt='np.linalg.inv(T1)', setup=transforms_setup, number=N)
    print(f"np.linalg.inv:       {t/N*1e6:.3g} μs")

if False:
    quat_setup = '''
import spatialmath.base as tr
import spatialmath.quaternion as qq
import numpy as np
q1 = tr.rand()
q2 = tr.rand()
v = np.r_[1,2,3]
Q1 = qq.UnitQuaternion.Rx(0.2)
Q2 = qq.UnitQuaternion.Ry(0.3)
'''
    t = timeit.timeit(stmt='a = tr.qqmul(q1,q2)', setup=quat_setup, number=N)
    print(f"quat.qqmul:         {t:.3g} μs")
    t = timeit.timeit(stmt='a = tr.qvmul(q1,v)', setup=quat_setup, number=N)
    print(f"quat.qqmul:         {t:.3g} μs")
    t = timeit.timeit(stmt='a = qq.UnitQuaternion()', setup=quat_setup, number=N)
    print(f"UnitQuaternion() :  {t:.3g} μs")
    t = timeit.timeit(stmt='a = qq.UnitQuaternion.Rx(0.2)', setup=quat_setup, number=N)
    print(f"UnitQuaternion.Rx : {t:.3g} μs")
    t = timeit.timeit(stmt='a = Q1 * Q2', setup=quat_setup, number=N)
    print(f"UnitQuaternion *:   {t:.3g} μs")
