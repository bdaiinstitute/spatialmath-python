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
            return np.r_[x, x ** 2, x * y ** 2]

        nt.assert_array_almost_equal(
            numjac(f, [2, 3]),
            np.array([[1, 0], [4, 0], [9, 12]]),  # x, 0  # 2x, 0  # y^2, 2xy
        )

        # test on rotation matrix
        nt.assert_array_almost_equal(numjac(rotx, [0], SO=3), np.array([[1, 0, 0]]).T)

        nt.assert_array_almost_equal(
            numjac(rotx, [pi / 2], SO=3), np.array([[1, 0, 0]]).T
        )

        nt.assert_array_almost_equal(numjac(roty, [0], SO=3), np.array([[0, 1, 0]]).T)

        nt.assert_array_almost_equal(numjac(rotz, [0], SO=3), np.array([[0, 0, 1]]).T)

    def test_rpy2jac(self):

        # ZYX order
        gamma = [0, 0, 0]
        nt.assert_array_almost_equal(rpy2jac(gamma), numjac(rpy2r, gamma, SO=3))
        gamma = [pi / 4, 0, -pi / 4]
        nt.assert_array_almost_equal(rpy2jac(gamma), numjac(rpy2r, gamma, SO=3))
        gamma = [-pi / 4, pi / 2, pi / 4]
        nt.assert_array_almost_equal(rpy2jac(gamma), numjac(rpy2r, gamma, SO=3))

        # XYZ order
        f = lambda gamma: rpy2r(gamma, order="xyz")
        gamma = [0, 0, 0]
        nt.assert_array_almost_equal(
            rpy2jac(gamma, order="xyz"), numjac(f, gamma, SO=3)
        )
        f = lambda gamma: rpy2r(gamma, order="xyz")
        gamma = [pi / 4, 0, -pi / 4]
        nt.assert_array_almost_equal(
            rpy2jac(gamma, order="xyz"), numjac(f, gamma, SO=3)
        )
        f = lambda gamma: rpy2r(gamma, order="xyz")
        gamma = [-pi / 4, pi / 2, pi / 4]
        nt.assert_array_almost_equal(
            rpy2jac(gamma, order="xyz"), numjac(f, gamma, SO=3)
        )

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

    # def test_rotvelxform(self):

    #     gamma = [0.1, 0.2, 0.3]
    #     R = rpy2r(gamma, order="zyx")
    #     A = rotvelxform(R, representation="rpy/zyx")
    #     self.assertEqual(A.shape, (6, 6))
    #     A3 = np.linalg.inv(A[3:6, 3:6])
    #     nt.assert_array_almost_equal(A3, rpy2jac(gamma, order="zyx"))

    #     gamma = [0.1, 0.2, 0.3]
    #     R = rpy2r(gamma, order="xyz")
    #     A = rot2jac(R, representation="rpy/xyz")
    #     self.assertEqual(A.shape, (6, 6))
    #     A3 = np.linalg.inv(A[3:6, 3:6])
    #     nt.assert_array_almost_equal(A3, rpy2jac(gamma, order="xyz"))

    #     gamma = [0.1, 0.2, 0.3]
    #     R = eul2r(gamma)
    #     A = rot2jac(R, representation="eul")
    #     self.assertEqual(A.shape, (6, 6))
    #     A3 = np.linalg.inv(A[3:6, 3:6])
    #     nt.assert_array_almost_equal(A3, eul2jac(gamma))

    #     gamma = [0.1, 0.2, 0.3]
    #     R = trexp(gamma)
    #     A = rot2jac(R, representation="exp")
    #     self.assertEqual(A.shape, (6, 6))
    #     A3 = np.linalg.inv(A[3:6, 3:6])
    #     nt.assert_array_almost_equal(A3, exp2jac(gamma))

    def test_rotvelxform(self):
        # compare inverse result against rpy/eul/exp2jac
        # compare forward and inverse results

        gamma = [0.1, 0.2, 0.3]
        A = rotvelxform(gamma, full=False, representation="rpy/zyx")
        Ai = rotvelxform(gamma, full=False, inverse=True, representation="rpy/zyx")
        nt.assert_array_almost_equal(A, rpy2jac(gamma, order="zyx"))
        nt.assert_array_almost_equal(Ai @ A, np.eye(3))

        gamma = [0.1, 0.2, 0.3]
        A = rotvelxform(gamma, full=False, representation="rpy/xyz")
        Ai = rotvelxform(gamma, full=False, inverse=True, representation="rpy/xyz")
        nt.assert_array_almost_equal(A, rpy2jac(gamma, order="xyz"))
        nt.assert_array_almost_equal(Ai @ A, np.eye(3))

        gamma = [0.1, 0.2, 0.3]
        A = rotvelxform(gamma, full=False, representation="eul")
        Ai = rotvelxform(gamma, full=False, inverse=True, representation="eul")
        nt.assert_array_almost_equal(A, eul2jac(gamma))
        nt.assert_array_almost_equal(Ai @ A, np.eye(3))

        gamma = [0.1, 0.2, 0.3]
        A = rotvelxform(gamma, full=False, representation="exp")
        Ai = rotvelxform(gamma, full=False, inverse=True, representation="exp")
        nt.assert_array_almost_equal(A, exp2jac(gamma))
        nt.assert_array_almost_equal(Ai @ A, np.eye(3))

    def test_rotvelxform_full(self):
        # compare inverse result against rpy/eul/exp2jac
        # compare forward and inverse results

        gamma = [0.1, 0.2, 0.3]
        A = rotvelxform(gamma, full=True, representation="rpy/zyx")
        Ai = rotvelxform(gamma, full=True, inverse=True, representation="rpy/zyx")
        nt.assert_array_almost_equal(A[3:,3:], rpy2jac(gamma, order="zyx"))
        nt.assert_array_almost_equal(A @ Ai, np.eye(6))

        gamma = [0.1, 0.2, 0.3]
        A = rotvelxform(gamma, full=True, representation="rpy/xyz")
        Ai = rotvelxform(gamma, full=True, inverse=True, representation="rpy/xyz")
        nt.assert_array_almost_equal(A[3:,3:], rpy2jac(gamma, order="xyz"))
        nt.assert_array_almost_equal(A @ Ai, np.eye(6))

        gamma = [0.1, 0.2, 0.3]
        A = rotvelxform(gamma, full=True, representation="eul")
        Ai = rotvelxform(gamma, full=True, inverse=True, representation="eul")
        nt.assert_array_almost_equal(A[3:,3:], eul2jac(gamma))
        nt.assert_array_almost_equal(A @ Ai, np.eye(6))

        gamma = [0.1, 0.2, 0.3]
        A = rotvelxform(gamma, full=True, representation="exp")
        Ai = rotvelxform(gamma, full=True, inverse=True, representation="exp")
        nt.assert_array_almost_equal(A[3:,3:], exp2jac(gamma))
        nt.assert_array_almost_equal(A @ Ai, np.eye(6))

    def test_angvelxform_inv_dot_eul(self):
        rep = 'eul'
        gamma = [0.1, 0.2, 0.3]
        gamma_d = [2, 3, 4]
        H = numhess(lambda g: rotvelxform(g, representation=rep, inverse=True, full=False), gamma)
        Adot = np.tensordot(H, gamma_d, (0, 0))
        res = rotvelxform_inv_dot(gamma, gamma_d, representation=rep, full=False)
        nt.assert_array_almost_equal(Adot, res, decimal=4)

    def test_angvelxform_dot_rpy_xyz(self):
        rep = 'rpy/xyz'
        gamma = [0.1, 0.2, 0.3]
        gamma_d = [2, 3, 4]
        H = numhess(lambda g: rotvelxform(g, representation=rep, inverse=True, full=False), gamma)
        Adot = np.zeros((3,3))
        Adot = np.tensordot(H, gamma_d, (0, 0))
        res = rotvelxform_inv_dot(gamma, gamma_d, representation=rep, full=False)
        nt.assert_array_almost_equal(Adot, res, decimal=4)

    def test_angvelxform_dot_rpy_zyx(self):
        rep = 'rpy/zyx'
        gamma = [0.1, 0.2, 0.3]
        gamma_d = [2, 3, 4]
        H = numhess(lambda g: rotvelxform(g, representation=rep, inverse=True, full=False), gamma)
        Adot = np.tensordot(H, gamma_d, (0, 0))
        res = rotvelxform_inv_dot(gamma, gamma_d, representation=rep, full=False)
        nt.assert_array_almost_equal(Adot, res, decimal=4)

    @unittest.skip("bug in angvelxform_dot for exponential coordinates")
    def test_angvelxform_dot_exp(self):
        rep = 'exp'
        gamma = [0.1, 0.2, 0.3]
        gamma_d = [2, 3, 4]
        H = numhess(lambda g: rotvelxform(g, representation=rep, inverse=True, full=False), gamma)
        Adot = np.tensordot(H, gamma_d, (0, 0))
        res = rotvelxform_inv_dot(gamma, gamma_d, representation=rep, full=False)
        nt.assert_array_almost_equal(Adot, res, decimal=4)

    def test_x_tr(self):
        # test transformation between pose and task-space vector representation

        T = transl(1, 2, 3) @ eul2tr((0.2, 0.3, 0.4))

        x = tr2x(T)
        nt.assert_array_almost_equal(x2tr(x), T)

        x = tr2x(T, representation='eul')
        nt.assert_array_almost_equal(x2tr(x, representation='eul'), T)

        x = tr2x(T, representation='rpy/xyz')
        nt.assert_array_almost_equal(x2tr(x, representation='rpy/xyz'), T)

        x = tr2x(T, representation='rpy/zyx')
        nt.assert_array_almost_equal(x2tr(x, representation='rpy/zyx'), T)

        x = tr2x(T, representation='exp')
        nt.assert_array_almost_equal(x2tr(x, representation='exp'), T)

        x = tr2x(T, representation='eul')
        nt.assert_array_almost_equal(x2tr(x, representation='eul'), T)

        x = tr2x(T, representation='arm')
        nt.assert_array_almost_equal(x2tr(x, representation='rpy/xyz'), T)

        x = tr2x(T, representation='vehicle')
        nt.assert_array_almost_equal(x2tr(x, representation='rpy/zyx'), T)

        x = tr2x(T, representation='exp')
        nt.assert_array_almost_equal(x2tr(x, representation='exp'), T)


    # def test_angvelxform_dot(self):

    #     gamma = [0.1, 0.2, 0.3]
    #     options = dict(full=False, representation='rpy/zyx')

    #     f = lambda gamma: angvelxform(gamma, options)

    #     nt.assert_array_almost_equal(angvelxform_dot(gamma, options), numjac(f))


# ---------------------------------------------------------------------------------------#
if __name__ == "__main__":

    unittest.main()
