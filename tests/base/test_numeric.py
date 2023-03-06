#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 14:19:04 2020

@author: corkep

"""

import numpy as np
import numpy.testing as nt
import unittest
import math


from spatialmath.base.numeric import *


class TestNumeric(unittest.TestCase):
    def test_numjac(self):

        pass

    def test_array2str(self):

        x = [1.2345678]
        s = array2str(x)

        self.assertIsInstance(s, str)
        self.assertEqual(s, "[ 1.23 ]")

        s = array2str(x, fmt="{:.5f}")
        self.assertEqual(s, "[ 1.23457 ]")

        s = array2str([1, 2, 3])
        self.assertEqual(s, "[ 1, 2, 3 ]")

        s = array2str([1, 2, 3], valuesep=":")
        self.assertEqual(s, "[ 1:2:3 ]")

        s = array2str([1, 2, 3], brackets=("<< ", " >>"))
        self.assertEqual(s, "<< 1, 2, 3 >>")

        s = array2str([1, 2e-8, 3])
        self.assertEqual(s, "[ 1, 2e-08, 3 ]")

        s = array2str([1, -2e-14, 3])
        self.assertEqual(s, "[ 1, 0, 3 ]")

        x = np.array([[1, 2, 3], [4, 5, 6]])
        s = array2str(x)
        self.assertEqual(s, "[ 1, 2, 3 | 4, 5, 6 ]")

    def test_bresenham(self):

        x, y = bresenham((-10, -10), (20, 10))
        self.assertIsInstance(x, np.ndarray)
        self.assertEqual(x.ndim, 1)
        self.assertIsInstance(y, np.ndarray)
        self.assertEqual(y.ndim, 1)
        self.assertEqual(len(x), len(y))

        # test points are no more than sqrt(2) apart
        z = np.array([x, y])
        d = np.diff(z, axis=1)
        d = np.linalg.norm(d, axis=0)
        self.assertTrue(all(d <= np.sqrt(2)))

        x, y = bresenham((20, 10), (-10, -10))

        # test points are no more than sqrt(2) apart
        z = np.array([x, y])
        d = np.diff(z, axis=1)
        d = np.linalg.norm(d, axis=0)
        self.assertTrue(all(d <= np.sqrt(2)))

        x, y = bresenham((-10, -10), (10, 20))

        # test points are no more than sqrt(2) apart
        z = np.array([x, y])
        d = np.diff(z, axis=1)
        d = np.linalg.norm(d, axis=0)
        self.assertTrue(all(d <= np.sqrt(2)))

        x, y = bresenham((10, 20), (-10, -10))

        # test points are no more than sqrt(2) apart
        z = np.array([x, y])
        d = np.diff(z, axis=1)
        d = np.linalg.norm(d, axis=0)
        self.assertTrue(all(d <= np.sqrt(2)))

    def test_mpq(self):

        data = np.array([[-1, 1, 1, -1], [-1, -1, 1, 1]])

        self.assertEqual(mpq_point(data, 0, 0), 4)
        self.assertEqual(mpq_point(data, 1, 0), 0)
        self.assertEqual(mpq_point(data, 0, 1), 0)

    def test_gauss1d(self):

        x = np.arange(-10, 10, 0.02)
        y = gauss1d(2, 1, x)

        self.assertEqual(len(x), len(y))

        m = np.argmax(y)
        self.assertAlmostEqual(x[m], 2)

    def test_gauss2d(self):

        r = np.arange(-10, 10, 0.02)
        X, Y = np.meshgrid(r, r)
        Z = gauss2d([2, 3], np.eye(2), X, Y)

        m = np.unravel_index(np.argmax(Z, axis=None), Z.shape)
        self.assertAlmostEqual(r[m[0]], 3)
        self.assertAlmostEqual(r[m[1]], 2)


# ---------------------------------------------------------------------------------------#
if __name__ == "__main__":

    unittest.main()
