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

from spatialmath.base.vectors import *
from spatialmath.base import sym
import matplotlib.pyplot as plt


class TestVector(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        plt.close("all")

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

        self.assertIsNone(unitvec([0, 0, 0]))
        self.assertIsNone(unitvec([0]))
        self.assertIsNone(unitvec(0))

    def test_colvec(self):

        t = np.r_[1, 2, 3]
        cv = colvec(t)
        self.assertEqual(cv.shape, (3, 1))
        nt.assert_array_almost_equal(cv.flatten(), t)

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
        self.assertAlmostEqual(norm([0, 0, 0]), 0)
        self.assertAlmostEqual(norm([1, 2, 3]), math.sqrt(14))
        self.assertAlmostEqual(norm(np.r_[1, 2, 3]), math.sqrt(14))

        x, y = sym.symbol("x y")
        v = [x, y]
        self.assertEqual(norm(v), sym.sqrt(x ** 2 + y ** 2))
        self.assertEqual(norm(np.r_[v]), sym.sqrt(x ** 2 + y ** 2))

    def test_norm(self):
        self.assertAlmostEqual(norm([0, 0, 0]), 0)
        self.assertAlmostEqual(normsq([1, 2, 3]), 14)
        self.assertAlmostEqual(normsq(np.r_[1, 2, 3]), 14)

        x, y = sym.symbol("x y")
        v = [x, y]
        self.assertEqual(normsq(v), x ** 2 + y ** 2)
        self.assertEqual(normsq(np.r_[v]), x ** 2 + y ** 2)

    def test_cross(self):

        A = np.eye(3)

        for i in range(0, 3):
            j = (i + 1) % 3
            k = (i + 2) % 3
            self.assertTrue(all(cross(A[:, i], A[:, j]) == A[:, k]))

    def test_isunittwist(self):
        # 3D
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

        # 2D
        # unit rotational twist
        self.assertTrue(isunittwist2([1, 2, 1]))

        # not a unit rotational twist
        self.assertFalse(isunittwist2([1, 2, 3]))

        # unit translation twist
        self.assertTrue(isunittwist2([1, 0, 0]))

        # not a unit translation twist
        self.assertFalse(isunittwist2([2, 0, 0]))

        with self.assertRaises(ValueError):
            isunittwist([3, 4])

        with self.assertRaises(ValueError):
            isunittwist2([3, 4])

    def test_unittwist(self):
        nt.assert_array_almost_equal(
            unittwist([0, 0, 0, 1, 0, 0]), np.r_[0, 0, 0, 1, 0, 0]
        )
        nt.assert_array_almost_equal(
            unittwist([0, 0, 0, 0, 2, 0]), np.r_[0, 0, 0, 0, 1, 0]
        )
        nt.assert_array_almost_equal(
            unittwist([0, 0, 0, 0, 0, -3]), np.r_[0, 0, 0, 0, 0, -1]
        )

        nt.assert_array_almost_equal(
            unittwist([1, 0, 0, 1, 0, 0]), np.r_[1, 0, 0, 1, 0, 0]
        )
        nt.assert_array_almost_equal(
            unittwist([1, 0, 0, 0, 2, 0]), np.r_[0.5, 0, 0, 0, 1, 0]
        )
        nt.assert_array_almost_equal(
            unittwist([1, 0, 0, 0, 0, -2]), np.r_[0.5, 0, 0, 0, 0, -1]
        )

        nt.assert_array_almost_equal(
            unittwist([1, 0, 0, 0, 0, 0]), np.r_[1, 0, 0, 0, 0, 0]
        )
        nt.assert_array_almost_equal(
            unittwist([0, 2, 0, 0, 0, 0]), np.r_[0, 1, 0, 0, 0, 0]
        )
        nt.assert_array_almost_equal(
            unittwist([0, 0, -2, 0, 0, 0]), np.r_[0, 0, -1, 0, 0, 0]
        )

        self.assertIsNone(unittwist([0, 0, 0, 0, 0, 0]))

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
        self.assertEqual(a, (None, None))

    def test_iszerovec(self):
        self.assertTrue(iszerovec([0]))
        self.assertTrue(iszerovec([0, 0]))
        self.assertTrue(iszerovec([0, 0, 0]))

        self.assertFalse(iszerovec([1]), False)
        self.assertFalse(iszerovec([0, 1]), False)
        self.assertFalse(iszerovec([0, 1, 0]), False)

    def test_iszero(self):
        self.assertTrue(iszero(0))
        self.assertFalse(iszero(1))

    def test_angdiff(self):

        self.assertEqual(angdiff(0, 0), 0)
        self.assertEqual(angdiff(np.pi, 0), -np.pi)
        self.assertEqual(angdiff(-np.pi, np.pi), 0)

    def test_removesmall(self):

        v = np.r_[1, 2, 3]
        nt.assert_array_almost_equal(removesmall(v), v)

        v = np.r_[1, 2, 3, 1e-6, -1e-6]
        nt.assert_array_almost_equal(removesmall(v), v)

        v = np.r_[1, 2, 3, 1e-15, -1e-15]
        nt.assert_array_almost_equal(removesmall(v), [1, 2, 3, 0, 0])

        v = np.r_[1, 2, 3, 1e-10, -1e-10]
        nt.assert_array_almost_equal(removesmall(v), [1, 2, 3, 1e-10, -1e-10])

        v = np.r_[1, 2, 3, 1e-10, -1e-10]
        nt.assert_array_almost_equal(removesmall(v, tol=1e8), [1, 2, 3, 0, 0])


# ---------------------------------------------------------------------------------------#
if __name__ == "__main__":

    unittest.main()
