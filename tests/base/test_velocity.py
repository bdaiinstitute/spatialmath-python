#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 14:19:04 2020

@author: corkep

"""


import numpy as np
import numpy.testing as nt
import unittest
from math import pi
import math
from scipy.linalg import logm, expm

from spatialmath.base.transforms3d import *
from spatialmath.base.numeric import *
from spatialmath.base.transformsNd import isR, t2r, r2t, rt2tr

import matplotlib.pyplot as plt


class TestVelocity(unittest.TestCase):

    def test_numjac(self):

        # test on algebraic example
        def f(X):
            x = X[0]
            y = X[1]
            return np.r_[x, x**2, x*y**2]

        nt.assert_array_almost_equal(numjac(f, [2, 3]),
            np.array([
                [1, 0],  # x, 0
                [4, 0],  # 2x, 0
                [9, 12]  # y^2, 2xy
            ]))

        # test on rotation matrix
        nt.assert_array_almost_equal(numjac(rotx, [0], SO=3),
            np.array([[1, 0, 0]]).T)

        nt.assert_array_almost_equal(numjac(rotx, [pi / 2], SO=3),
            np.array([[1, 0, 0]]).T)

        nt.assert_array_almost_equal(numjac(roty, [0], SO=3),
            np.array([[0, 1, 0]]).T)

        nt.assert_array_almost_equal(numjac(rotz, [0], SO=3),
            np.array([[0, 0, 1]]).T)

    def test_rpy2jac(self):

        # ZYX order
        gamma = [0, 0, 0]
        nt.assert_array_almost_equal(rpy2jac(gamma), numjac(rpy2r, gamma, SO=3))
        gamma = [pi / 4, 0, -pi / 4]
        nt.assert_array_almost_equal(rpy2jac(gamma), numjac(rpy2r, gamma, SO=3))
        gamma = [-pi / 4, pi / 2, pi / 4]
        nt.assert_array_almost_equal(rpy2jac(gamma), numjac(rpy2r, gamma, SO=3))

        # XYZ order
        f = lambda gamma: rpy2r(gamma, order='xyz')
        gamma = [0, 0, 0]
        nt.assert_array_almost_equal(rpy2jac(gamma, order='xyz'), numjac(f, gamma, SO=3))
        f = lambda gamma: rpy2r(gamma, order='xyz')
        gamma = [pi / 4, 0, -pi / 4]
        nt.assert_array_almost_equal(rpy2jac(gamma, order='xyz'), numjac(f, gamma, SO=3))
        f = lambda gamma: rpy2r(gamma, order='xyz')
        gamma = [-pi / 4, pi / 2, pi / 4]
        nt.assert_array_almost_equal(rpy2jac(gamma, order='xyz'), numjac(f, gamma, SO=3))


    def test_eul2jac(self):

        # ZYX order
        gamma = [0, 0, 0]
        nt.assert_array_almost_equal(eul2jac(gamma), numjac(eul2r, gamma, SO=3))
        gamma = [pi / 4, 0, -pi / 4]
        nt.assert_array_almost_equal(eul2jac(gamma), numjac(eul2r, gamma, SO=3))
        gamma = [-pi / 4, pi / 2, pi / 4]
        nt.assert_array_almost_equal(eul2jac(gamma), numjac(eul2r, gamma, SO=3))

    def test_exp2jac(self):

        # ZYX order
        gamma = np.r_[1, 0, 0]
        nt.assert_array_almost_equal(exp2jac(gamma), numjac(exp2r, gamma, SO=3))
        gamma = np.r_[0.2, 0.3, 0.4]
        nt.assert_array_almost_equal(exp2jac(gamma), numjac(exp2r, gamma, SO=3))
        gamma = np.r_[0, 0, 0]
        nt.assert_array_almost_equal(exp2jac(gamma), numjac(exp2r, gamma, SO=3))

    def test_rot2jac(self):

        gamma = [0.1, 0.2, 0.3]
        R = rpy2r(gamma, order='zyx')
        A = rot2jac(R, representation='rpy/zyx')
        self.assertEqual(A.shape, (6,6))
        A3 = np.linalg.inv(A[3:6,3:6])
        nt.assert_array_almost_equal(A3, rpy2jac(gamma, order='zyx'))

        gamma = [0.1, 0.2, 0.3]
        R = rpy2r(gamma, order='xyz')
        A = rot2jac(R, representation='rpy/xyz')
        self.assertEqual(A.shape, (6,6))
        A3 = np.linalg.inv(A[3:6,3:6])
        nt.assert_array_almost_equal(A3, rpy2jac(gamma, order='xyz'))

        gamma = [0.1, 0.2, 0.3]
        R = eul2r(gamma)
        A = rot2jac(R, representation='eul')
        self.assertEqual(A.shape, (6,6))
        A3 = np.linalg.inv(A[3:6,3:6])
        nt.assert_array_almost_equal(A3, eul2jac(gamma))


    def test_angvelxform(self):

        gamma = [0.1, 0.2, 0.3]
        A = angvelxform(gamma, full=False, representation='rpy/zyx')
        Ai = angvelxform(gamma, full=False, inverse=True, representation='rpy/zyx')
        nt.assert_array_almost_equal(Ai, rpy2jac(gamma, order='zyx'))
        nt.assert_array_almost_equal(A @ Ai, np.eye(3))

        gamma = [0.1, 0.2, 0.3]
        A = angvelxform(gamma, full=False, representation='rpy/xyz')
        Ai = angvelxform(gamma, full=False, inverse=True, representation='rpy/xyz')
        nt.assert_array_almost_equal(Ai, rpy2jac(gamma, order='xyz'))
        nt.assert_array_almost_equal(A @ Ai, np.eye(3))

        gamma = [0.1, 0.2, 0.3]
        A = angvelxform(gamma, full=False, representation='eul')
        Ai = angvelxform(gamma, full=False, inverse=True, representation='eul')
        nt.assert_array_almost_equal(Ai, eul2jac(gamma))
        nt.assert_array_almost_equal(A @ Ai, np.eye(3))

    # def test_angvelxform_dot(self):

    #     gamma = [0.1, 0.2, 0.3]
    #     options = dict(full=False, representation='rpy/zyx')

    #     f = lambda gamma: angvelxform(gamma, options)

    #     nt.assert_array_almost_equal(angvelxform_dot(gamma, options), numjac(f))
# ---------------------------------------------------------------------------------------#
if __name__ == '__main__':

    unittest.main()
