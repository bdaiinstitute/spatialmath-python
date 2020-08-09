#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 14:19:04 2020

@author: corkep

"""

# This file is part of the SpatialMath toolbox for Python
# https://github.com/petercorke/spatialmath-python
# 
# MIT License
# 
# Copyright (c) 1993-2020 Peter Corke
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Contributors:
# 
#     1. Luis Fernando Lara Tobar and Peter Corke, 2008
#     2. Josh Carrigg Hodson, Aditya Dua, Chee Ho Chan, 2017 (robopy)
#     3. Peter Corke, 2020

# Some unit tests

import numpy.testing as nt
import unittest
from math import pi
from scipy.linalg import logm, expm

from spatialmath.base import *

import matplotlib.pyplot as plt


class TestVector(unittest.TestCase):

    def test_unit(self):

        nt.assert_array_almost_equal(unitvec([1, 0, 0]), np.r_[1, 0, 0])
        nt.assert_array_almost_equal(unitvec([0, 1, 0]), np.r_[0, 1, 0])
        nt.assert_array_almost_equal(unitvec([0, 0, 1]), np.r_[0, 0, 1])

        nt.assert_array_almost_equal(unitvec((1, 0, 0)), np.r_[1, 0, 0])
        nt.assert_array_almost_equal(unitvec((0, 1, 0)), np.r_[0, 1, 0])
        nt.assert_array_almost_equal(unitvec((0, 0, 1)), np.r_[0, 0, 1])

        nt.assert_array_almost_equal(unitvec(np.r_[1, 0, 0]), np.r_[1, 0, 0])
        nt.assert_array_almost_equal(unitvec(np.r_[0, 1, 0]), np.r_[0, 1, 0])
        nt.assert_array_almost_equal(unitvec(np.r_[0, 0, 1]), np.r_[0, 0, 1])

        nt.assert_array_almost_equal(unitvec([9, 0, 0]), np.r_[1, 0, 0])
        nt.assert_array_almost_equal(unitvec([0, 9, 0]), np.r_[0, 1, 0])
        nt.assert_array_almost_equal(unitvec([0, 0, 9]), np.r_[0, 0, 1])
        
        self.assertIsNone(unitvec([0, 0, 0]) )
        self.assertIsNone(unitvec([0]) )
        self.assertIsNone(unitvec(0) )

    def test_isunitvec(self):
        self.assertTrue(isunitvec([1, 0, 0]))
        self.assertTrue(isunitvec((1, 0, 0)))
        self.assertTrue(isunitvec(np.r_[1, 0, 0]))

        self.assertFalse(isunitvec([9, 0, 0]))
        self.assertFalse(isunitvec((9, 0, 0)))
        self.assertFalse(isunitvec(np.r_[9, 0, 0]))
        
        self.assertTrue(isunitvec(1))
        self.assertTrue(isunitvec([1]))
        self.assertTrue(isunitvec(-1))
        self.assertTrue(isunitvec([-1]))
        
        self.assertFalse(isunitvec(2))
        self.assertFalse(isunitvec([2]))
        self.assertFalse(isunitvec(-2))
        self.assertFalse(isunitvec([-2]))
        

    def test_norm(self):
        nt.assert_array_almost_equal(norm([0, 0, 0]), 0)
        nt.assert_array_almost_equal(norm([1, 2, 3]), math.sqrt(14))

    def test_isunittwist(self):
        # unit rotational twist
        self.assertTrue(isunittwist([1, 2, 3, 1, 0, 0]))
        self.assertTrue(isunittwist((1, 2, 3, 1, 0, 0)))
        self.assertTrue(isunittwist(np.r_[1, 2, 3, 1, 0, 0]))

        # not a unit rotational twist
        self.assertFalse(isunittwist([1, 2, 3, 1, 0, 1]))

        # unit translation twist
        self.assertTrue(isunittwist([1, 0, 0, 0, 0, 0]))

        # not a unit translation twist
        self.assertFalse(isunittwist([2, 0, 0, 0, 0, 0]))
        
    def test_unittwist(self):
        nt.assert_array_almost_equal(unittwist([0, 0, 0, 1, 0, 0]), np.r_[0, 0, 0, 1, 0, 0])
        nt.assert_array_almost_equal(unittwist([0, 0, 0, 0, 2, 0]), np.r_[0, 0, 0, 0, 1, 0])
        nt.assert_array_almost_equal(unittwist([0, 0, 0, 0, 0, -3]), np.r_[0, 0, 0, 0, 0, -1])
        
        nt.assert_array_almost_equal(unittwist([1, 0, 0, 1, 0, 0]), np.r_[1, 0, 0, 1, 0, 0])
        nt.assert_array_almost_equal(unittwist([1, 0, 0, 0, 2, 0]), np.r_[0.5, 0, 0, 0, 1, 0])
        nt.assert_array_almost_equal(unittwist([1, 0, 0, 0, 0, -2]), np.r_[0.5, 0, 0, 0, 0, -1])
        
        nt.assert_array_almost_equal(unittwist([1, 0, 0, 0, 0, 0]), np.r_[1, 0, 0, 0, 0, 0])
        nt.assert_array_almost_equal(unittwist([0, 2, 0, 0, 0, 0]), np.r_[0, 1, 0, 0, 0, 0])
        nt.assert_array_almost_equal(unittwist([0, 0, -2, 0, 0, 0]), np.r_[0, 0, -1, 0, 0, 0])

        self.assertIsNone(unittwist([0, 0, 0, 0, 0, 0]) )

    def test_unittwist_norm(self):
        a = unittwist_norm([0, 0, 0, 1, 0, 0])
        nt.assert_array_almost_equal(a[0], np.r_[0, 0, 0, 1, 0, 0])
        nt.assert_array_almost_equal(a[1], 1)
        
        a = unittwist_norm([0, 0, 0, 0, 2, 0])
        nt.assert_array_almost_equal(a[0], np.r_[0, 0, 0, 0, 1, 0])
        nt.assert_array_almost_equal(a[1], 2)
        
        a = unittwist_norm([0, 0, 0, 0, 0, -3])
        nt.assert_array_almost_equal(a[0], np.r_[0, 0, 0, 0, 0, -1])
        nt.assert_array_almost_equal(a[1], 3)
        
        a = unittwist_norm([1, 0, 0, 1, 0, 0])
        nt.assert_array_almost_equal(a[0], np.r_[1, 0, 0, 1, 0, 0])
        nt.assert_array_almost_equal(a[1], 1)
        
        a = unittwist_norm([1, 0, 0, 0, 2, 0])
        nt.assert_array_almost_equal(a[0], np.r_[0.5, 0, 0, 0, 1, 0])
        nt.assert_array_almost_equal(a[1], 2)
        
        a = unittwist_norm([1, 0, 0, 0, 0, -2])
        nt.assert_array_almost_equal(a[0], np.r_[0.5, 0, 0, 0, 0, -1])
        nt.assert_array_almost_equal(a[1], 2)
        
        a = unittwist_norm([1, 0, 0, 0, 0, 0])
        nt.assert_array_almost_equal(a[0], np.r_[1, 0, 0, 0, 0, 0])
        nt.assert_array_almost_equal(a[1], 1)
        
        a = unittwist_norm([0, 2, 0, 0, 0, 0])
        nt.assert_array_almost_equal(a[0], np.r_[0, 1, 0, 0, 0, 0])
        nt.assert_array_almost_equal(a[1], 2)
        
        a = unittwist_norm([0, 0, -2, 0, 0, 0])
        nt.assert_array_almost_equal(a[0], np.r_[0, 0, -1, 0, 0, 0])
        nt.assert_array_almost_equal(a[1], 2)

        a = unittwist_norm([0, 0, 0, 0, 0, 0])
        self.assertEqual(a, (None, None) )

    def test_iszerovec(self):
        self.assertTrue(iszerovec([0]))
        self.assertTrue(iszerovec([0, 0]))
        self.assertTrue(iszerovec([0, 0, 0]))

        self.assertFalse(iszerovec([1]), False)
        self.assertFalse(iszerovec([0, 1]), False)
        self.assertFalse(iszerovec([0, 1, 0]), False)


class TestND(unittest.TestCase):
    def test_iseye(self):
        self.assertTrue(iseye(np.eye(1)))
        self.assertTrue(iseye(np.eye(2)))
        self.assertTrue(iseye(np.eye(3)))
        self.assertTrue(iseye(np.eye(5)))

        self.assertFalse(iseye(2 * np.eye(3)))
        self.assertFalse(iseye(-np.eye(3)))
        self.assertFalse(iseye(np.array([[1, 0, 0], [0, 1, 0]])))
        self.assertFalse(iseye(np.array([1, 0, 0])))

    def test_Rt(self):
        nt.assert_array_almost_equal(rotx(0.3), t2r(trotx(0.3)))
        nt.assert_array_almost_equal(trotx(0.3), r2t(rotx(0.3)))

        R = rotx(0.2)
        t = [3, 4, 5]
        T = rt2tr(R, t)
        nt.assert_array_almost_equal(t2r(T), R)
        nt.assert_array_almost_equal(transl(T), np.array(t))

    def test_checks(self):

        # 3D case, with rotation matrix
        R = np.eye(3)
        self.assertTrue(isR(R))
        self.assertFalse(isrot2(R))
        self.assertTrue(isrot(R))
        self.assertFalse(ishom(R))
        self.assertTrue(ishom2(R))
        self.assertFalse(isrot2(R, True))
        self.assertTrue(isrot(R, True))
        self.assertFalse(ishom(R, True))
        self.assertTrue(ishom2(R, True))

        # 3D case, invalid rotation matrix
        R = np.eye(3)
        R[0, 1] = 2
        self.assertFalse(isR(R))
        self.assertFalse(isrot2(R))
        self.assertTrue(isrot(R))
        self.assertFalse(ishom(R))
        self.assertTrue(ishom2(R))
        self.assertFalse(isrot2(R, True))
        self.assertFalse(isrot(R, True))
        self.assertFalse(ishom(R, True))
        self.assertFalse(ishom2(R, True))

        # 3D case, with rotation matrix
        T = np.array([[1, 0, 0, 3], [0, 1, 0, 4], [0, 0, 1, 5], [0, 0, 0, 1]])
        self.assertFalse(isR(T))
        self.assertFalse(isrot2(T))
        self.assertFalse(isrot(T))
        self.assertTrue(ishom(T))
        self.assertFalse(ishom2(T))
        self.assertFalse(isrot2(T, True))
        self.assertFalse(isrot(T, True))
        self.assertTrue(ishom(T, True))
        self.assertFalse(ishom2(T, True))

        # 3D case, invalid rotation matrix
        T = np.array([[1, 0, 0, 3], [0, 1, 1, 4], [0, 0, 1, 5], [0, 0, 0, 1]])
        self.assertFalse(isR(T))
        self.assertFalse(isrot2(T))
        self.assertFalse(isrot(T))
        self.assertTrue(ishom(T),)
        self.assertFalse(ishom2(T))
        self.assertFalse(isrot2(T, True))
        self.assertFalse(isrot(T, True))
        self.assertFalse(ishom(T, True))
        self.assertFalse(ishom2(T, True))

        # 3D case, invalid bottom row
        T = np.array([[1, 0, 0, 3], [0, 1, 1, 4], [0, 0, 1, 5], [9, 0, 0, 1]])
        self.assertFalse(isR(T))
        self.assertFalse(isrot2(T))
        self.assertFalse(isrot(T))
        self.assertTrue(ishom(T))
        self.assertFalse(ishom2(T))
        self.assertFalse(isrot2(T, True))
        self.assertFalse(isrot(T, True))
        self.assertFalse(ishom(T, True))
        self.assertFalse(ishom2(T, True))

        # skew matrices
        S = np.array([
            [0, 2],
            [-2, 0]])
        nt.assert_equal(isskew(S), True)
        S[0, 0] = 1
        nt.assert_equal(isskew(S), False)

        S = np.array([
            [0, -3, 2],
            [3, 0, -1],
            [-2, 1, 0]])
        nt.assert_equal(isskew(S), True)
        S[0, 0] = 1
        nt.assert_equal(isskew(S), False)

        # augmented skew matrices
        S = np.array([
            [0, 2, 3],
            [-2, 0, 4],
            [0, 0, 0]])
        nt.assert_equal(isskewa(S), True)
        S[0, 0] = 1
        nt.assert_equal(isskew(S), False)
        S[0, 0] = 0
        S[2, 0] = 1
        nt.assert_equal(isskew(S), False)

    def test_homog(self):
        nt.assert_almost_equal(e2h([1, 2, 3]), np.r_[1, 2, 3, 1])

        nt.assert_almost_equal(h2e([2, 4, 6, 2]), np.r_[1, 2, 3])


class Test2D(unittest.TestCase):
    def test_rot2(self):
        R = np.array([[1, 0], [0, 1]])
        nt.assert_array_almost_equal(rot2(0), R)
        nt.assert_array_almost_equal(rot2(0, unit='rad'), R)
        nt.assert_array_almost_equal(rot2(0, unit='deg'), R)
        nt.assert_array_almost_equal(rot2(0, 'deg'), R)
        nt.assert_almost_equal(np.linalg.det(rot2(0)), 1)

        R = np.array([[0, -1], [1, 0]])
        nt.assert_array_almost_equal(rot2(pi / 2), R)
        nt.assert_array_almost_equal(rot2(pi / 2, unit='rad'), R)
        nt.assert_array_almost_equal(rot2(90, unit='deg'), R)
        nt.assert_array_almost_equal(rot2(90, 'deg'), R)
        nt.assert_almost_equal(np.linalg.det(rot2(pi / 2)), 1)

        R = np.array([[-1, 0], [0, -1]])
        nt.assert_array_almost_equal(rot2(pi), R)
        nt.assert_array_almost_equal(rot2(pi, unit='rad'), R)
        nt.assert_array_almost_equal(rot2(180, unit='deg'), R)
        nt.assert_array_almost_equal(rot2(180, 'deg'), R)
        nt.assert_almost_equal(np.linalg.det(rot2(pi)), 1)

    def test_trot2(self):
        nt.assert_array_almost_equal(trot2(pi / 2, t=[3, 4]), np.array([[0, -1, 3], [1, 0, 4], [0, 0, 1]]))
        nt.assert_array_almost_equal(trot2(pi / 2, t=(3, 4)), np.array([[0, -1, 3], [1, 0, 4], [0, 0, 1]]))
        nt.assert_array_almost_equal(trot2(pi / 2, t=np.array([3, 4])), np.array([[0, -1, 3], [1, 0, 4], [0, 0, 1]]))

    def test_Rt(self):
        nt.assert_array_almost_equal(rot2(0.3), t2r(trot2(0.3)))
        nt.assert_array_almost_equal(trot2(0.3), r2t(rot2(0.3)))

        R = rot2(0.2)
        t = [1, 2]
        T = rt2tr(R, t)
        nt.assert_array_almost_equal(t2r(T), R)
        nt.assert_array_almost_equal(transl2(T), np.array(t))
        # TODO

    def test_transl2(self):
        nt.assert_array_almost_equal(transl2(1, 2), np.array([[1, 0, 1], [0, 1, 2], [0, 0, 1]]))
        nt.assert_array_almost_equal(transl2([1, 2]), np.array([[1, 0, 1], [0, 1, 2], [0, 0, 1]]))

    def test_print2(self):

        T = transl2(1, 2) @ trot2(0.3)

        s = trprint2(T, file=None)
        self.assertIsInstance(s, str)
        self.assertEqual(len(s), 36)

    def test_checks(self):
        # 2D case, with rotation matrix
        R = np.eye(2)
        nt.assert_equal(isR(R), True)
        nt.assert_equal(isrot2(R), True)
        nt.assert_equal(isrot(R), False)
        nt.assert_equal(ishom(R), False)
        nt.assert_equal(ishom2(R), False)
        nt.assert_equal(isrot2(R, True), True)
        nt.assert_equal(isrot(R, True), False)
        nt.assert_equal(ishom(R, True), False)
        nt.assert_equal(ishom2(R, True), False)

        # 2D case, invalid rotation matrix
        R = np.array([[1, 1], [0, 1]])
        nt.assert_equal(isR(R), False)
        nt.assert_equal(isrot2(R), True)
        nt.assert_equal(isrot(R), False)
        nt.assert_equal(ishom(R), False)
        nt.assert_equal(ishom2(R), False)
        nt.assert_equal(isrot2(R, True), False)
        nt.assert_equal(isrot(R, True), False)
        nt.assert_equal(ishom(R, True), False)
        nt.assert_equal(ishom2(R, True), False)

        # 2D case, with homogeneous transformation matrix
        T = np.array([[1, 0, 3], [0, 1, 4], [0, 0, 1]])
        nt.assert_equal(isR(T), False)
        nt.assert_equal(isrot2(T), False)
        nt.assert_equal(isrot(T), True)
        nt.assert_equal(ishom(T), False)
        nt.assert_equal(ishom2(T), True)
        nt.assert_equal(isrot2(T, True), False)
        nt.assert_equal(isrot(T, True), False)
        nt.assert_equal(ishom(T, True), False)
        nt.assert_equal(ishom2(T, True), True)

        # 2D case, invalid rotation matrix
        T = np.array([[1, 1, 3], [0, 1, 4], [0, 0, 1]])
        nt.assert_equal(isR(T), False)
        nt.assert_equal(isrot2(T), False)
        nt.assert_equal(isrot(T), True)
        nt.assert_equal(ishom(T), False)
        nt.assert_equal(ishom2(T), True)
        nt.assert_equal(isrot2(T, True), False)
        nt.assert_equal(isrot(T, True), False)
        nt.assert_equal(ishom(T, True), False)
        nt.assert_equal(ishom2(T, True), False)

        # 2D case, invalid bottom row
        T = np.array([[1, 1, 3], [0, 1, 4], [9, 0, 1]])
        nt.assert_equal(isR(T), False)
        nt.assert_equal(isrot2(T), False)
        nt.assert_equal(isrot(T), True)
        nt.assert_equal(ishom(T), False)
        nt.assert_equal(ishom2(T), True)
        nt.assert_equal(isrot2(T, True), False)
        nt.assert_equal(isrot(T, True), False)
        nt.assert_equal(ishom(T, True), False)
        nt.assert_equal(ishom2(T, True), False)
        
    
    def test_trinterp2(self):

        T0 = trot2(-0.3)
        T1 = trot2(0.3)
        
        nt.assert_array_almost_equal( trinterp2(start=T0, end=T1, s=0), T0)
        nt.assert_array_almost_equal( trinterp2(start=T0, end=T1, s=1), T1)
        nt.assert_array_almost_equal( trinterp2(start=T0, end=T1, s=0.5), np.eye(3))
        
        T0 = transl2(-1, -2)
        T1 = transl2(1, 2)
        
        nt.assert_array_almost_equal( trinterp2(start=T0, end=T1, s=0), T0)
        nt.assert_array_almost_equal( trinterp2(start=T0, end=T1, s=1), T1)
        nt.assert_array_almost_equal( trinterp2(start=T0, end=T1, s=0.5), np.eye(3))
        
        T0 = transl2(-1, -2) @ trot2(-0.3)
        T1 = transl2(1, 2) @ trot2(0.3)
        
        nt.assert_array_almost_equal( trinterp2(start=T0, end=T1, s=0), T0)
        nt.assert_array_almost_equal( trinterp2(start=T0, end=T1, s=1), T1)
        nt.assert_array_almost_equal( trinterp2(start=T0, end=T1, s=0.5), np.eye(3))
        
        nt.assert_array_almost_equal( trinterp2(start=T0, end=T1, s=0), T0)
        nt.assert_array_almost_equal( trinterp2(start=T0, end=T1, s=1), T1)
        nt.assert_array_almost_equal( trinterp2(start=T0, end=T1, s=0.5), np.eye(3))

    def test_plot(self):
        plt.figure()
        trplot2(transl2(1, 2), block=False, frame='A', rviz=True, width=1)
        trplot2(transl2(3, 1), block=False, color='red', arrow=True, width=3, frame='B')
        trplot2(transl2(4, 3)@trot2(math.pi / 3), block=False, color='green', frame='c')


class Test3D(unittest.TestCase):

    def test_trinv(self):
        T = np.eye(4)
        nt.assert_array_almost_equal(trinv(T), T)
        
        T = trotx(0.3)
        nt.assert_array_almost_equal(trinv(T)@T, np.eye(4))
        
        T = transl(1,2,3)
        nt.assert_array_almost_equal(trinv(T)@T, np.eye(4))
        
    def test_rotx(self):
        R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        nt.assert_array_almost_equal(rotx(0), R)
        nt.assert_array_almost_equal(rotx(0, unit='rad'), R)
        nt.assert_array_almost_equal(rotx(0, unit='deg'), R)
        nt.assert_array_almost_equal(rotx(0, 'deg'), R)
        nt.assert_almost_equal(np.linalg.det(rotx(0)), 1)

        R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        nt.assert_array_almost_equal(rotx(pi / 2), R)
        nt.assert_array_almost_equal(rotx(pi / 2, unit='rad'), R)
        nt.assert_array_almost_equal(rotx(90, unit='deg'), R)
        nt.assert_array_almost_equal(rotx(90, 'deg'), R)
        nt.assert_almost_equal(np.linalg.det(rotx(pi / 2)), 1)

        R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        nt.assert_array_almost_equal(rotx(pi), R)
        nt.assert_array_almost_equal(rotx(pi, unit='rad'), R)
        nt.assert_array_almost_equal(rotx(180, unit='deg'), R)
        nt.assert_array_almost_equal(rotx(180, 'deg'), R)
        nt.assert_almost_equal(np.linalg.det(rotx(pi)), 1)

    def test_roty(self):
        R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        nt.assert_array_almost_equal(roty(0), R)
        nt.assert_array_almost_equal(roty(0, unit='rad'), R)
        nt.assert_array_almost_equal(roty(0, unit='deg'), R)
        nt.assert_array_almost_equal(roty(0, 'deg'), R)
        nt.assert_almost_equal(np.linalg.det(roty(0)), 1)

        R = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        nt.assert_array_almost_equal(roty(pi / 2), R)
        nt.assert_array_almost_equal(roty(pi / 2, unit='rad'), R)
        nt.assert_array_almost_equal(roty(90, unit='deg'), R)
        nt.assert_array_almost_equal(roty(90, 'deg'), R)
        nt.assert_almost_equal(np.linalg.det(roty(pi / 2)), 1)

        R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        nt.assert_array_almost_equal(roty(pi), R)
        nt.assert_array_almost_equal(roty(pi, unit='rad'), R)
        nt.assert_array_almost_equal(roty(180, unit='deg'), R)
        nt.assert_array_almost_equal(roty(180, 'deg'), R)
        nt.assert_almost_equal(np.linalg.det(roty(pi)), 1)

    def test_rotz(self):
        R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        nt.assert_array_almost_equal(rotz(0), R)
        nt.assert_array_almost_equal(rotz(0, unit='rad'), R)
        nt.assert_array_almost_equal(rotz(0, unit='deg'), R)
        nt.assert_array_almost_equal(rotz(0, 'deg'), R)
        nt.assert_almost_equal(np.linalg.det(rotz(0)), 1)

        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        nt.assert_array_almost_equal(rotz(pi / 2), R)
        nt.assert_array_almost_equal(rotz(pi / 2, unit='rad'), R)
        nt.assert_array_almost_equal(rotz(90, unit='deg'), R)
        nt.assert_array_almost_equal(rotz(90, 'deg'), R)
        nt.assert_almost_equal(np.linalg.det(rotz(pi / 2)), 1)

        R = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        nt.assert_array_almost_equal(rotz(pi), R)
        nt.assert_array_almost_equal(rotz(pi, unit='rad'), R)
        nt.assert_array_almost_equal(rotz(180, unit='deg'), R)
        nt.assert_array_almost_equal(rotz(180, 'deg'), R)
        nt.assert_almost_equal(np.linalg.det(rotz(pi)), 1)

    def test_trotX(self):
        T = np.array([[1, 0, 0, 3], [0, 0, -1, 4], [0, 1, 0, 5], [0, 0, 0, 1]])
        nt.assert_array_almost_equal(trotx(pi / 2, t=[3, 4, 5]), T)
        nt.assert_array_almost_equal(trotx(pi / 2, t=(3, 4, 5)), T)
        nt.assert_array_almost_equal(trotx(pi / 2, t=np.array([3, 4, 5])), T)

        T = np.array([[0, 0, 1, 3], [0, 1, 0, 4], [-1, 0, 0, 5], [0, 0, 0, 1]])
        nt.assert_array_almost_equal(troty(pi / 2, t=[3, 4, 5]), T)
        nt.assert_array_almost_equal(troty(pi / 2, t=(3, 4, 5)), T)
        nt.assert_array_almost_equal(troty(pi / 2, t=np.array([3, 4, 5])), T)

        T = np.array([[0, -1, 0, 3], [1, 0, 0, 4], [0, 0, 1, 5], [0, 0, 0, 1]])
        nt.assert_array_almost_equal(trotz(pi / 2, t=[3, 4, 5]), T)
        nt.assert_array_almost_equal(trotz(pi / 2, t=(3, 4, 5)), T)
        nt.assert_array_almost_equal(trotz(pi / 2, t=np.array([3, 4, 5])), T)

    def test_rpy2r(self):

        r2d = 180 / pi

        # default zyx order
        R = rotz(0.3) @ roty(0.2) @ rotx(0.1)
        nt.assert_array_almost_equal(rpy2r(0.1, 0.2, 0.3), R)
        nt.assert_array_almost_equal(rpy2r([0.1, 0.2, 0.3]), R)
        nt.assert_array_almost_equal(rpy2r(0.1 * r2d, 0.2 * r2d, 0.3 * r2d, unit='deg'), R)
        nt.assert_array_almost_equal(rpy2r([0.1 * r2d, 0.2 * r2d, 0.3 * r2d], unit='deg'), R)

        # xyz order
        R = rotx(0.3) @ roty(0.2) @ rotz(0.1)
        nt.assert_array_almost_equal(rpy2r(0.1, 0.2, 0.3, order='xyz'), R)
        nt.assert_array_almost_equal(rpy2r([0.1, 0.2, 0.3], order='xyz'), R)
        nt.assert_array_almost_equal(rpy2r(0.1 * r2d, 0.2 * r2d, 0.3 * r2d, unit='deg', order='xyz'), R)
        nt.assert_array_almost_equal(rpy2r([0.1 * r2d, 0.2 * r2d, 0.3 * r2d], unit='deg', order='xyz'), R)

        # yxz order
        R = roty(0.3) @ rotx(0.2) @ rotz(0.1)
        nt.assert_array_almost_equal(rpy2r(0.1, 0.2, 0.3, order='yxz'), R)
        nt.assert_array_almost_equal(rpy2r([0.1, 0.2, 0.3], order='yxz'), R)
        nt.assert_array_almost_equal(rpy2r(0.1 * r2d, 0.2 * r2d, 0.3 * r2d, unit='deg', order='yxz'), R)
        nt.assert_array_almost_equal(rpy2r([0.1 * r2d, 0.2 * r2d, 0.3 * r2d], unit='deg', order='yxz'), R)

    def test_rpy2tr(self):

        r2d = 180 / pi

        # default zyx order
        T = trotz(0.3) @ troty(0.2) @ trotx(0.1)
        nt.assert_array_almost_equal(rpy2tr(0.1, 0.2, 0.3), T)
        nt.assert_array_almost_equal(rpy2tr([0.1, 0.2, 0.3]), T)
        nt.assert_array_almost_equal(rpy2tr(0.1 * r2d, 0.2 * r2d, 0.3 * r2d, unit='deg'), T)
        nt.assert_array_almost_equal(rpy2tr([0.1 * r2d, 0.2 * r2d, 0.3 * r2d], unit='deg'), T)

        # xyz order
        T = trotx(0.3) @ troty(0.2) @ trotz(0.1)
        nt.assert_array_almost_equal(rpy2tr(0.1, 0.2, 0.3, order='xyz'), T)
        nt.assert_array_almost_equal(rpy2tr([0.1, 0.2, 0.3], order='xyz'), T)
        nt.assert_array_almost_equal(rpy2tr(0.1 * r2d, 0.2 * r2d, 0.3 * r2d, unit='deg', order='xyz'), T)
        nt.assert_array_almost_equal(rpy2tr([0.1 * r2d, 0.2 * r2d, 0.3 * r2d], unit='deg', order='xyz'), T)

        # yxz order
        T = troty(0.3) @ trotx(0.2) @ trotz(0.1)
        nt.assert_array_almost_equal(rpy2tr(0.1, 0.2, 0.3, order='yxz'), T)
        nt.assert_array_almost_equal(rpy2tr([0.1, 0.2, 0.3], order='yxz'), T)
        nt.assert_array_almost_equal(rpy2tr(0.1 * r2d, 0.2 * r2d, 0.3 * r2d, unit='deg', order='yxz'), T)
        nt.assert_array_almost_equal(rpy2tr([0.1 * r2d, 0.2 * r2d, 0.3 * r2d], unit='deg', order='yxz'), T)

    def test_eul2r(self):

        r2d = 180 / pi

        # default zyx order
        R = rotz(0.1) @ roty(0.2) @ rotz(0.3)
        nt.assert_array_almost_equal(eul2r(0.1, 0.2, 0.3), R)
        nt.assert_array_almost_equal(eul2r([0.1, 0.2, 0.3]), R)
        nt.assert_array_almost_equal(eul2r(0.1 * r2d, 0.2 * r2d, 0.3 * r2d, unit='deg'), R)
        nt.assert_array_almost_equal(eul2r([0.1 * r2d, 0.2 * r2d, 0.3 * r2d], unit='deg'), R)

    def test_eul2tr(self):

        r2d = 180 / pi

        # default zyx order
        T = trotz(0.1) @ troty(0.2) @ trotz(0.3)
        nt.assert_array_almost_equal(eul2tr(0.1, 0.2, 0.3), T)
        nt.assert_array_almost_equal(eul2tr([0.1, 0.2, 0.3]), T)
        nt.assert_array_almost_equal(eul2tr(0.1 * r2d, 0.2 * r2d, 0.3 * r2d, unit='deg'), T)
        nt.assert_array_almost_equal(eul2tr([0.1 * r2d, 0.2 * r2d, 0.3 * r2d], unit='deg'), T)

    def test_tr2rpy(self):
        rpy = np.r_[0.1, 0.2, 0.3]
        R = rpy2r(rpy)
        nt.assert_array_almost_equal(tr2rpy(R), rpy)
        nt.assert_array_almost_equal(tr2rpy(R, unit='deg'), rpy * 180 / pi)

        T = rpy2tr(rpy)
        nt.assert_array_almost_equal(tr2rpy(T), rpy,)
        nt.assert_array_almost_equal(tr2rpy(T, unit='deg'), rpy * 180 / pi)

        # xyz order
        R = rpy2r(rpy, order='xyz')
        nt.assert_array_almost_equal(tr2rpy(R, order='xyz'), rpy)
        nt.assert_array_almost_equal(tr2rpy(R, unit='deg', order='xyz'), rpy * 180 / pi)

        T = rpy2tr(rpy, order='xyz')
        nt.assert_array_almost_equal(tr2rpy(T, order='xyz'), rpy)
        nt.assert_array_almost_equal(tr2rpy(T, unit='deg', order='xyz'), rpy * 180 / pi)

        # corner cases
        seq = 'zyx'
        ang = [pi, 0, 0]
        a = rpy2tr(ang, order=seq)
        nt.assert_array_almost_equal(rpy2tr(tr2rpy(a, order=seq), order=seq), a)
        ang = [0, pi, 0]
        a = rpy2tr(ang, order=seq)
        nt.assert_array_almost_equal(rpy2tr(tr2rpy(a, order=seq), order=seq), a)
        ang = [0, 0, pi]
        a = rpy2tr(ang, order=seq)
        nt.assert_array_almost_equal(rpy2tr(tr2rpy(a, order=seq), order=seq), a)
        ang = [0, pi / 2, 0]  # singularity
        a = rpy2tr(ang, order=seq)
        nt.assert_array_almost_equal(rpy2tr(tr2rpy(a, order=seq), order=seq), a)
        ang = [0, -pi / 2, 0]
        a = rpy2tr(ang, order=seq)
        nt.assert_array_almost_equal(rpy2tr(tr2rpy(a, order=seq), order=seq), a)

        seq = 'xyz'
        ang = [pi, 0, 0]
        a = rpy2tr(ang, order=seq)
        nt.assert_array_almost_equal(rpy2tr(tr2rpy(a, order=seq), order=seq), a)
        ang = [0, pi, 0]
        a = rpy2tr(ang, order=seq)
        nt.assert_array_almost_equal(rpy2tr(tr2rpy(a, order=seq), order=seq), a)
        ang = [0, 0, pi]
        a = rpy2tr(ang, order=seq)
        nt.assert_array_almost_equal(rpy2tr(tr2rpy(a, order=seq), order=seq), a)
        ang = [0, pi / 2, 0]  # singularity
        a = rpy2tr(ang, order=seq)
        nt.assert_array_almost_equal(rpy2tr(tr2rpy(a, order=seq), order=seq), a)
        ang = [0, -pi / 2, 0]
        a = rpy2tr(ang, order=seq)
        nt.assert_array_almost_equal(rpy2tr(tr2rpy(a, order=seq), order=seq), a)

        seq = 'yxz'
        ang = [pi, 0, 0]
        a = rpy2tr(ang, order=seq)
        nt.assert_array_almost_equal(rpy2tr(tr2rpy(a, order=seq), order=seq), a)
        ang = [0, pi, 0]
        a = rpy2tr(ang, order=seq)
        nt.assert_array_almost_equal(rpy2tr(tr2rpy(a, order=seq), order=seq), a)
        ang = [0, 0, pi]
        a = rpy2tr(ang, order=seq)
        nt.assert_array_almost_equal(rpy2tr(tr2rpy(a, order=seq), order=seq), a)
        ang = [0, pi / 2, 0]  # singularity
        a = rpy2tr(ang, order=seq)
        nt.assert_array_almost_equal(rpy2tr(tr2rpy(a, order=seq), order=seq), a)
        ang = [0, -pi / 2, 0]
        a = rpy2tr(ang, order=seq)
        nt.assert_array_almost_equal(rpy2tr(tr2rpy(a, order=seq), order=seq), a)

    def test_tr2eul(self):

        eul = np.r_[0.1, 0.2, 0.3]
        R = eul2r(eul)
        nt.assert_array_almost_equal(tr2eul(R), eul)
        nt.assert_array_almost_equal(tr2eul(R, unit='deg'), eul * 180 / pi)

        T = eul2tr(eul)
        nt.assert_array_almost_equal(tr2eul(T), eul)
        nt.assert_array_almost_equal(tr2eul(T, unit='deg'), eul * 180 / pi)

        # test singularity case
        eul = [0.1, 0, 0.3]
        R = eul2r(eul)
        nt.assert_array_almost_equal(eul2r(tr2eul(R)), R)
        nt.assert_array_almost_equal(eul2r(tr2eul(R, unit='deg'), unit='deg'), R)

        # test flip
        eul = [-0.1, 0.2, 0.3]
        R = eul2r(eul)
        eul2 = tr2eul(R, flip=True)
        nt.assert_equal(eul2[0] > 0, True)
        nt.assert_array_almost_equal(eul2r(eul2), R)

    def test_tr2angvec(self):

        # null rotation
        # - vector isn't defined here, but RTB sets it (0 0 0)
        [theta, v] = tr2angvec(np.eye(3, 3))
        nt.assert_array_almost_equal(theta, 0.0)
        nt.assert_array_almost_equal(v, np.r_[0, 0, 0])

        # canonic rotations
        [theta, v] = tr2angvec(rotx(pi / 2))
        nt.assert_array_almost_equal(theta, pi / 2)
        nt.assert_array_almost_equal(v, np.r_[1, 0, 0])

        [theta, v] = tr2angvec(roty(pi / 2))
        nt.assert_array_almost_equal(theta, pi / 2)
        nt.assert_array_almost_equal(v, np.r_[0, 1, 0])

        [theta, v] = tr2angvec(rotz(pi / 2))
        nt.assert_array_almost_equal(theta, pi / 2)
        nt.assert_array_almost_equal(v, np.r_[0, 0, 1])

        # null rotation
        [theta, v] = tr2angvec(np.eye(4))
        nt.assert_array_almost_equal(theta, 0.0)
        nt.assert_array_almost_equal(v, np.r_[0, 0, 0])

        # canonic rotations
        [theta, v] = tr2angvec(trotx(pi / 2))
        nt.assert_array_almost_equal(theta, pi / 2)
        nt.assert_array_almost_equal(v, np.r_[1, 0, 0])

        [theta, v] = tr2angvec(troty(pi / 2))
        nt.assert_array_almost_equal(theta, pi / 2)
        nt.assert_array_almost_equal(v, np.r_[0, 1, 0])

        [theta, v] = tr2angvec(trotz(pi / 2))
        nt.assert_array_almost_equal(theta, pi / 2)
        nt.assert_array_almost_equal(v, np.r_[0, 0, 1])

        [theta, v] = tr2angvec(roty(pi / 2), unit='deg')
        nt.assert_array_almost_equal(theta, 90)
        nt.assert_array_almost_equal(v, np.r_[0, 1, 0])

    def test_print(self):

        R = rotx(0.3) @  roty(0.4)
        s = trprint(R, file=None)
        self.assertIsInstance(s, str)
        self.assertEqual(len(s), 43)

        T = transl(1, 2, 3) @ trotx(0.3) @  troty(0.4)
        s = trprint(T, file=None)
        self.assertIsInstance(s, str)
        self.assertEqual(len(s), 76)
        self.assertTrue('rpy' in s)
        self.assertTrue('zyx' in s)

        s = trprint(T, file=None, orient='rpy/xyz')
        self.assertIsInstance(s, str)
        self.assertEqual(len(s), 76)
        self.assertTrue('rpy' in s)
        self.assertTrue('xyz' in s)

        s = trprint(T, file=None, orient='eul')
        self.assertIsInstance(s, str)
        self.assertEqual(len(s), 72)
        self.assertTrue('eul' in s)
        self.assertFalse('zyx' in s)

    def test_plot(self):
        plt.figure()
        trplot(transl(1, 2, 3), block=False, frame='A', rviz=True, width=1, dims=[0, 10, 0, 10, 0, 10])
        trplot(transl(3, 1, 2), block=False, color='red', width=3, frame='B')
        trplot(transl(4, 3, 1)@trotx(math.pi / 3), block=False, color='green', frame='c', dims=[0, 4, 0, 4, 0, 4])

        plt.clf()
        tranimate(transl(1, 2, 3), repeat=False, pause=2)
        # run again, with axes already created
        tranimate(transl(1, 2, 3), repeat=False, pause=2, dims=[0, 10, 0, 10, 0, 10])

        # test animate with line not arrow, text, test with SO(3)

    def test_trinterp(self):
        T0 = trotx(-0.3)
        T1 = trotx(0.3)
        
        nt.assert_array_almost_equal( trinterp(start=T0, end=T1, s=0), T0)
        nt.assert_array_almost_equal( trinterp(start=T0, end=T1, s=1), T1)
        nt.assert_array_almost_equal( trinterp(start=T0, end=T1, s=0.5), np.eye(4))
        
        T0 = transl(-1, -2, -3)
        T1 = transl(1, 2, 3)
        
        nt.assert_array_almost_equal( trinterp(start=T0, end=T1, s=0), T0)
        nt.assert_array_almost_equal( trinterp(start=T0, end=T1, s=1), T1)
        nt.assert_array_almost_equal( trinterp(start=T0, end=T1, s=0.5), np.eye(4))
        
        T0 = transl(-1, -2, -3) @ trotx(-0.3)
        T1 = transl(1, 2, 3) @ trotx(0.3)
        
        nt.assert_array_almost_equal( trinterp(start=T0, end=T1, s=0), T0)
        nt.assert_array_almost_equal( trinterp(start=T0, end=T1, s=1), T1)
        nt.assert_array_almost_equal( trinterp(start=T0, end=T1, s=0.5), np.eye(4))
        
        nt.assert_array_almost_equal( trinterp(start=T0, end=T1, s=0), T0)
        nt.assert_array_almost_equal( trinterp(start=T0, end=T1, s=1), T1)
        nt.assert_array_almost_equal( trinterp(start=T0, end=T1, s=0.5), np.eye(4))
        
    def test_tr2delta(self):

        # unit testing tr2delta with a tr matrix
        nt.assert_array_almost_equal( tr2delta( transl(0.1, 0.2, 0.3) ), np.r_[0.1, 0.2, 0.3, 0, 0, 0])
        nt.assert_array_almost_equal( tr2delta( transl(0.1, 0.2, 0.3), transl(0.2, 0.4, 0.6) ), np.r_[0.1, 0.2, 0.3, 0, 0, 0])
        nt.assert_array_almost_equal( tr2delta( trotx(0.001) ), np.r_[0,0,0, 0.001,0,0])
        nt.assert_array_almost_equal( tr2delta( troty(0.001) ), np.r_[0,0,0, 0,0.001,0])
        nt.assert_array_almost_equal( tr2delta( trotz(0.001) ), np.r_[0,0,0, 0,0,0.001])
        nt.assert_array_almost_equal( tr2delta( trotx(0.001), trotx(0.002) ), np.r_[0,0,0, 0.001,0,0])
    
        # %Testing with a scalar number input
        # verifyError(tc, @()tr2delta(1),'SMTB:tr2delta:badarg');
        # verifyError(tc, @()tr2delta( ones(3,3) ),'SMTB:tr2delta:badarg');
        
    def test_delta2tr(self):
        # test with standard numbers  
        nt.assert_array_almost_equal(delta2tr([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]), 
        np.array([[1.0, -0.6, 0.5, 0.1], 
                  [0.6, 1.0, -0.4, 0.2], 
                  [-0.5, 0.4, 1.0, 0.3], 
                  [0, 0, 0, 1.0]]))

        # test, with, zeros
        nt.assert_array_almost_equal(delta2tr([0, 0, 0, 0, 0, 0]), np.eye(4));
        
        # test with scalar input 
        #verifyError(testCase, @()delta2tr(1),'MATLAB:badsubscript');
    
    def test_tr2jac(self):

        # NOTE, create these matrices using pyprint() in MATLAB
        nt.assert_array_almost_equal( tr2jac(trotx(pi/2)), 
            np.array([  [1, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0],
                        [0, -1, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, -1, 0]]))
         
        nt.assert_array_almost_equal( tr2jac(trotx(pi/2), True), 
            np.array([  [1, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0],
                        [0, -1, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, -1, 0]]))
        
        nt.assert_array_almost_equal( tr2jac(transl(1,2,3)),
            np.array([  [1, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1]]))
            
        nt.assert_array_almost_equal( tr2jac(transl(1,2,3), True),
            np.array([  [1, 0, 0, 0, 3, -2],
                        [0, 1, 0, -3, 0, 1],
                        [0, 0, 1, 2, -1, 0],
                        [0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1]]))

        # test with scalar value
        #verifyError(tc, @()tr2jac(1),'SMTB:t2r:badarg');

class TestLie(unittest.TestCase):

    def test_vex(self):
        S = np.array([
            [0, -3],
            [3, 0]
        ])

        nt.assert_array_almost_equal(vex(S), np.array([3]))
        nt.assert_array_almost_equal(vex(-S), np.array([-3]))

        S = np.array([
            [0, -3, 2],
            [3, 0, -1],
            [-2, 1, 0]
        ])

        nt.assert_array_almost_equal(vex(S), np.array([1, 2, 3]))
        nt.assert_array_almost_equal(vex(-S), -np.array([1, 2, 3]))

    def test_skew(self):
        R = skew(3)
        nt.assert_equal(isrot2(R, check=False), True)  # check size
        nt.assert_array_almost_equal(np.linalg.norm(R.T + R), 0)  # check is skew
        nt.assert_array_almost_equal(vex(R), np.array([3]))  # check contents, vex already verified

        R = skew([1, 2, 3])
        nt.assert_equal(isrot(R, check=False), True)  # check size
        nt.assert_array_almost_equal(np.linalg.norm(R.T + R), 0)  # check is skew
        nt.assert_array_almost_equal(vex(R), np.array([1, 2, 3]))  # check contents, vex already verified

    def test_vexa(self):

        S = np.array([
            [0, -3, 1],
            [3, 0, 2],
            [0, 0, 0]
        ])
        nt.assert_array_almost_equal(vexa(S), np.array([1, 2, 3]))

        S = np.array([
            [0, 3, -1],
            [-3, 0, 2],
            [0, 0, 0]
        ])
        nt.assert_array_almost_equal(vexa(S), np.array([-1, 2, -3]))

        S = np.array([
            [0, -6, 5, 1],
            [6, 0, -4, 2],
            [-5, 4, 0, 3],
            [0, 0, 0, 0]
        ])
        nt.assert_array_almost_equal(vexa(S), np.array([1, 2, 3, 4, 5, 6]))

        S = np.array([
            [0, 6, 5, 1],
            [-6, 0, 4, -2],
            [-5, -4, 0, 3],
            [0, 0, 0, 0]
        ])
        nt.assert_array_almost_equal(vexa(S), np.array([1, -2, 3, -4, 5, -6]))

    def test_skewa(self):
        T = skewa([3, 4, 5])
        nt.assert_equal(ishom2(T, check=False), True)  # check size
        R = t2r(T)
        nt.assert_equal(np.linalg.norm(R.T + R), 0)  # check is skew
        nt.assert_array_almost_equal(vexa(T), np.array([3, 4, 5]))  # check contents, vexa already verified

        T = skewa([1, 2, 3, 4, 5, 6])
        nt.assert_equal(ishom(T, check=False), True)  # check size
        R = t2r(T)
        nt.assert_equal(np.linalg.norm(R.T + R), 0)  # check is skew
        nt.assert_array_almost_equal(vexa(T), np.array([1, 2, 3, 4, 5, 6]))  # check contents, vexa already verified

    def test_trlog(self):

        # %%% SO(3) tests
        # zero rotation case
        nt.assert_array_almost_equal(trlog(np.eye(3)), skew([0, 0, 0]))

        # rotation by pi case
        nt.assert_array_almost_equal(trlog(rotx(pi)), skew([pi, 0, 0]))
        nt.assert_array_almost_equal(trlog(roty(pi)), skew([0, pi, 0]))
        nt.assert_array_almost_equal(trlog(rotz(pi)), skew([0, 0, pi]))

        # general case
        nt.assert_array_almost_equal(trlog(rotx(0.2)), skew([0.2, 0, 0]))
        nt.assert_array_almost_equal(trlog(roty(0.3)), skew([0, 0.3, 0]))
        nt.assert_array_almost_equal(trlog(rotz(0.4)), skew([0, 0, 0.4]))

        R = rotx(0.2) @ roty(0.3) @ rotz(0.4)
        nt.assert_array_almost_equal(trlog(R), logm(R))

        # %% SE(3) tests

        # pure translation
        nt.assert_array_almost_equal(trlog(transl([1, 2, 3])), np.array([[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 3], [0, 0, 0, 0]]))

        # pure rotation
        # rotation by pi case
        nt.assert_array_almost_equal(trlog(trotx(pi)), skewa([0, 0, 0, pi, 0, 0]))
        nt.assert_array_almost_equal(trlog(troty(pi)), skewa([0, 0, 0, 0, pi, 0]))
        nt.assert_array_almost_equal(trlog(trotz(pi)), skewa([0, 0, 0, 0, 0, pi]))

        # general case
        nt.assert_array_almost_equal(trlog(trotx(0.2)), skewa([0, 0, 0, 0.2, 0, 0]))
        nt.assert_array_almost_equal(trlog(troty(0.3)), skewa([0, 0, 0, 0, 0.3, 0]))
        nt.assert_array_almost_equal(trlog(trotz(0.4)), skewa([0, 0, 0, 0, 0, 0.4]))

        # mixture
        T = transl([1, 2, 3]) @ trotx(0.3)
        nt.assert_array_almost_equal(trlog(T), logm(T))

        T = transl([1, 2, 3]) @ troty(0.3)
        nt.assert_array_almost_equal(trlog(T), logm(T))

    # def test_trlog2(self):

    #     #%%% SO(2) tests
    #     # zero rotation case
    #     nt.assert_array_almost_equal(trlog2( np.eye(2) ), skew([0]))

    #     # rotation by pi case
    #     nt.assert_array_almost_equal(trlog2( rot2(pi) ), skew([pi]))

    #     # general case
    #     nt.assert_array_almost_equal(trlog2( rotx(0.2) ), skew([0.2]))

    #     #%% SE(3) tests

    #     # pure translation
    #     nt.assert_array_almost_equal(trlog2( transl2([1, 2]) ), np.array([[0, 0, 1], [ 0, 0, 2], [ 0, 0, 0]]))

    #     # pure rotation
    #     # rotation by pi case
    #     nt.assert_array_almost_equal(trlog( trot2(pi) ), skewa([0, 0, pi]))

    #     # general case
    #     nt.assert_array_almost_equal(trlog( trot2(0.2) ), skewa([0, 0, 0.2]))

    #     # mixture
    #     T = transl([1, 2, 3]) @ trot2(0.3)
    #     nt.assert_array_almost_equal(trlog2(T), logm(T))
    # TODO

    def test_trexp(self):

        # %% SO(3) tests

        # % so(3)

        # zero rotation case
        nt.assert_array_almost_equal(trexp(skew([0, 0, 0])), np.eye(3))
        nt.assert_array_almost_equal(trexp([0, 0, 0]), np.eye(3))

        # % so(3), theta

        # rotation by pi case
        nt.assert_array_almost_equal(trexp(skew([pi, 0, 0])), rotx(pi))
        nt.assert_array_almost_equal(trexp(skew([0, pi, 0])), roty(pi))
        nt.assert_array_almost_equal(trexp(skew([0, 0, pi])), rotz(pi))

        # general case
        nt.assert_array_almost_equal(trexp(skew([0.2, 0, 0])), rotx(0.2))
        nt.assert_array_almost_equal(trexp(skew([0, 0.3, 0])), roty(0.3))
        nt.assert_array_almost_equal(trexp(skew([0, 0, 0.4])), rotz(0.4))

        nt.assert_array_almost_equal(trexp(skew([1, 0, 0]), 0.2), rotx(0.2))
        nt.assert_array_almost_equal(trexp(skew([0, 1, 0]), 0.3), roty(0.3))
        nt.assert_array_almost_equal(trexp(skew([0, 0, 1]), 0.4), rotz(0.4))

        nt.assert_array_almost_equal(trexp([1, 0, 0], 0.2), rotx(0.2))
        nt.assert_array_almost_equal(trexp([0, 1, 0], 0.3), roty(0.3))
        nt.assert_array_almost_equal(trexp([0, 0, 1], 0.4), rotz(0.4))

        nt.assert_array_almost_equal(trexp(np.r_[1, 0, 0] * 0.2), rotx(0.2))
        nt.assert_array_almost_equal(trexp(np.r_[0, 1, 0] * 0.3), roty(0.3))
        nt.assert_array_almost_equal(trexp(np.r_[0, 0, 1] * 0.4), rotz(0.4))

        # %% SE(3) tests

        # zero motion case
        nt.assert_array_almost_equal(trexp(skewa([0, 0, 0, 0, 0, 0])), np.eye(4))
        nt.assert_array_almost_equal(trexp([0, 0, 0, 0, 0, 0]), np.eye(4))
        
        # % sigma = se(3)
        # pure translation
        nt.assert_array_almost_equal(trexp(skewa([1, 2, 3, 0, 0, 0])), transl([1, 2, 3]))
        nt.assert_array_almost_equal(trexp(skewa([0, 0, 0, 0.2, 0, 0])), trotx(0.2))
        nt.assert_array_almost_equal(trexp(skewa([0, 0, 0, 0, 0.3, 0])), troty(0.3))
        nt.assert_array_almost_equal(trexp(skewa([0, 0, 0, 0, 0, 0.4])), trotz(0.4))

        nt.assert_array_almost_equal(trexp([1, 2, 3, 0, 0, 0]), transl([1, 2, 3]))
        nt.assert_array_almost_equal(trexp([0, 0, 0, 0.2, 0, 0]), trotx(0.2))
        nt.assert_array_almost_equal(trexp([0, 0, 0, 0, 0.3, 0]), troty(0.3))
        nt.assert_array_almost_equal(trexp([0, 0, 0, 0, 0, 0.4]), trotz(0.4))

        # mixture
        S = skewa([1, 2, 3, 0.1, -0.2, 0.3])
        nt.assert_array_almost_equal(trexp(S), expm(S))

        # twist vector
        #nt.assert_array_almost_equal(trexp( double(Twist(T))), T)

        # (sigma, theta)
        nt.assert_array_almost_equal(trexp(skewa([1, 0, 0, 0, 0, 0]), 2), transl([2, 0, 0]))
        nt.assert_array_almost_equal(trexp(skewa([0, 1, 0, 0, 0, 0]), 2), transl([0, 2, 0]))
        nt.assert_array_almost_equal(trexp(skewa([0, 0, 1, 0, 0, 0]), 2), transl([0, 0, 2]))

        nt.assert_array_almost_equal(trexp(skewa([0, 0, 0, 1, 0, 0]), 0.2), trotx(0.2))
        nt.assert_array_almost_equal(trexp(skewa([0, 0, 0, 0, 1, 0]), 0.2), troty(0.2))
        nt.assert_array_almost_equal(trexp(skewa([0, 0, 0, 0, 0, 1]), 0.2), trotz(0.2))

        # (twist, theta)
        #nt.assert_array_almost_equal(trexp(Twist('R', [1, 0, 0], [0, 0, 0]).S, 0.3), trotx(0.3))

        T = transl([1, 2, 3])@trotz(0.3)
        nt.assert_array_almost_equal(trexp(trlog(T)), T)

    def test_trexp2(self):


        # % so(2)

        # zero rotation case
        nt.assert_array_almost_equal(trexp2(skew([0])), np.eye(2))
        nt.assert_array_almost_equal(trexp2(skew(0)), np.eye(2))

        # % so(2), theta

        # rotation by pi case
        nt.assert_array_almost_equal(trexp2(skew(pi)), rot2(pi))

        # general case
        nt.assert_array_almost_equal(trexp2(skew(0.2)), rot2(0.2))

        nt.assert_array_almost_equal(trexp2(1, 0.2), rot2(0.2))

        # %% SE(3) tests

        # % sigma = se(3)
        # pure translation
        nt.assert_array_almost_equal(trexp2(skewa([1, 2, 0])), transl2([1, 2]))

        nt.assert_array_almost_equal(trexp2([0, 0, 0.2]), trot2(0.2))

        # mixture
        S = skewa([1, 2, 0.3])
        nt.assert_array_almost_equal(trexp2(S), expm(S))

        # twist vector
        #nt.assert_array_almost_equal(trexp( double(Twist(T))), T)

        # (sigma, theta)
        nt.assert_array_almost_equal(trexp2(skewa([1, 0, 0]), 2), transl2([2, 0]))
        nt.assert_array_almost_equal(trexp2(skewa([0, 1, 0]), 2), transl2([0, 2]))

        nt.assert_array_almost_equal(trexp2(skewa([0, 0, 1]), 0.2), trot2(0.2))

        # (twist, theta)
        #nt.assert_array_almost_equal(trexp(Twist('R', [1, 0, 0], [0, 0, 0]).S, 0.3), trotx(0.3))

        # T = transl2([1, 2])@trot2(0.3)
        # nt.assert_array_almost_equal(trexp2(trlog2(T)), T)
        # TODO
        
    def test_trnorm(self):
        T0 = transl(-1, -2, -3) @ trotx(-0.3)
        nt.assert_array_almost_equal(trnorm(T0), T0)
        
        



# ---------------------------------------------------------------------------------------#
if __name__ == '__main__':

    unittest.main()
