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

try:
    import sympy as sp

    from spatialmath.base.symbolic import *

    _symbolics = True
except ImportError:
    _symbolics = False
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt

from math import pi


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

        with self.assertRaises(ValueError):
            unitvec([0, 0, 0])
        with self.assertRaises(ValueError):
            unitvec([0])
        with self.assertRaises(ValueError):
            unitvec(0)

    def test_colvec(self):
        t = np.r_[1, 2, 3]
        cv = colvec(t)
        assert cv.shape == (3, 1)
        nt.assert_array_almost_equal(cv.flatten(), t)

    def test_isunitvec(self):
        assert isunitvec([1, 0, 0])
        assert isunitvec((1, 0, 0))
        assert isunitvec(np.r_[1, 0, 0])

        assert not isunitvec([9, 0, 0])
        assert not isunitvec((9, 0, 0))
        assert not isunitvec(np.r_[9, 0, 0])

        assert isunitvec(1)
        assert isunitvec([1])
        assert isunitvec(-1)
        assert isunitvec([-1])

        assert not isunitvec(2)
        assert not isunitvec([2])
        assert not isunitvec(-2)
        assert not isunitvec([-2])

    def test_norm(self):
        self.assertAlmostEqual(norm([0, 0, 0]), 0)
        self.assertAlmostEqual(norm([1, 2, 3]), math.sqrt(14))
        self.assertAlmostEqual(norm(np.r_[1, 2, 3]), math.sqrt(14))

    def test_normsq(self):
        self.assertAlmostEqual(normsq([0, 0, 0]), 0)
        self.assertAlmostEqual(normsq([1, 2, 3]), 14)
        self.assertAlmostEqual(normsq(np.r_[1, 2, 3]), 14)

    @unittest.skipUnless(_symbolics, "sympy required")
    def test_norm_sym(self):
        x, y = symbol("x y")
        v = [x, y]
        assert norm(v) == sqrt(x**2 + y**2)
        assert norm(np.r_[v]) == sqrt(x**2 + y**2)

    def test_cross(self):
        A = np.eye(3)

        for i in range(0, 3):
            j = (i + 1) % 3
            k = (i + 2) % 3
            assert all(cross(A[:, i], A[:, j]) == A[:, k])

    def test_isunittwist(self):
        # 3D
        # unit rotational twist
        assert isunittwist([1, 2, 3, 1, 0, 0])
        assert isunittwist((1, 2, 3, 1, 0, 0))
        assert isunittwist(np.r_[1, 2, 3, 1, 0, 0])

        # not a unit rotational twist
        assert not isunittwist([1, 2, 3, 1, 0, 1])

        # unit translation twist
        assert isunittwist([1, 0, 0, 0, 0, 0])

        # not a unit translation twist
        assert not isunittwist([2, 0, 0, 0, 0, 0])

        # 2D
        # unit rotational twist
        assert isunittwist2([1, 2, 1])

        # not a unit rotational twist
        assert not isunittwist2([1, 2, 3])

        # unit translation twist
        assert isunittwist2([1, 0, 0])

        # not a unit translation twist
        assert not isunittwist2([2, 0, 0])

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

        assert unittwist([0, 0, 0, 0, 0, 0]) is None

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
        assert a[0] is None
        assert a[1] is None

    def test_unittwist2(self):
        nt.assert_array_almost_equal(
            unittwist2([1, 0, 0]), np.r_[1, 0, 0]
        )
        nt.assert_array_almost_equal(
            unittwist2([0, 2, 0]), np.r_[0, 1, 0]
        )
        nt.assert_array_almost_equal(
            unittwist2([0, 0, -3]), np.r_[0, 0, -1]
        )
        nt.assert_array_almost_equal(
            unittwist2([2, 0, -2]), np.r_[1, 0, -1]
        )

        assert unittwist2([0, 0, 0]) is None

    def test_unittwist2_norm(self):
        a = unittwist2_norm([1, 0, 0])
        nt.assert_array_almost_equal(a[0], np.r_[1, 0, 0])
        nt.assert_array_almost_equal(a[1], 1)

        a = unittwist2_norm([0, 2, 0])
        nt.assert_array_almost_equal(a[0], np.r_[0, 1, 0])
        nt.assert_array_almost_equal(a[1], 2)

        a = unittwist2_norm([0, 0, -3])
        nt.assert_array_almost_equal(a[0], np.r_[0, 0, -1])
        nt.assert_array_almost_equal(a[1], 3)

        a = unittwist2_norm([2, 0, -2])
        nt.assert_array_almost_equal(a[0], np.r_[1, 0, -1])
        nt.assert_array_almost_equal(a[1], 2)

        a = unittwist2_norm([0, 0, 0])
        assert a[0] is None
        assert a[1] is None

    def test_iszerovec(self):
        assert iszerovec([0])
        assert iszerovec([0, 0])
        assert iszerovec([0, 0, 0])

        assert not iszerovec([1]), False
        assert not iszerovec([0, 1]), False
        assert not iszerovec([0, 1, 0]), False

    def test_iszero(self):
        assert iszero(0)
        assert not iszero(1)

    def test_angdiff(self):
        assert angdiff(0, 0) == 0
        assert isinstance(angdiff(0, 0), float)
        assert angdiff(pi, 0) == -pi
        assert angdiff(-pi, pi) == 0

        x = angdiff([0, -pi, pi], 0)
        nt.assert_array_almost_equal(x, [0, -pi, -pi])
        assert isinstance(x, np.ndarray)
        nt.assert_array_almost_equal(angdiff([0, -pi, pi], pi), [-pi, 0, 0])

        x = angdiff(0, [0, -pi, pi])
        nt.assert_array_almost_equal(x, [0, -pi, -pi])
        assert isinstance(x, np.ndarray)
        nt.assert_array_almost_equal(angdiff(pi, [0, -pi, pi]), [-pi, 0, 0])

        x = angdiff([1, 2, 3], [1, 2, 3])
        nt.assert_array_almost_equal(x, [0, 0, 0])
        assert isinstance(x, np.ndarray)

    def test_wrap(self):
        self.assertAlmostEqual(wrap_0_2pi(0), 0)
        self.assertAlmostEqual(wrap_0_2pi(2 * pi), 0)
        self.assertAlmostEqual(wrap_0_2pi(3 * pi), pi)
        self.assertAlmostEqual(wrap_0_2pi(-pi), pi)
        nt.assert_array_almost_equal(
            wrap_0_2pi([0, 2 * pi, 3 * pi, -pi]), [0, 0, pi, pi]
        )

        self.assertAlmostEqual(wrap_mpi_pi(0), 0)
        self.assertAlmostEqual(wrap_mpi_pi(-pi), -pi)
        self.assertAlmostEqual(wrap_mpi_pi(pi), -pi)
        self.assertAlmostEqual(wrap_mpi_pi(2 * pi), 0)
        self.assertAlmostEqual(wrap_mpi_pi(1.5 * pi), -0.5 * pi)
        self.assertAlmostEqual(wrap_mpi_pi(-1.5 * pi), 0.5 * pi)
        nt.assert_array_almost_equal(
            wrap_mpi_pi([0, -pi, pi, 2 * pi, 1.5 * pi, -1.5 * pi]),
            [0, -pi, -pi, 0, -0.5 * pi, 0.5 * pi],
        )

        self.assertAlmostEqual(wrap_0_pi(0), 0)
        self.assertAlmostEqual(wrap_0_pi(pi), pi)
        self.assertAlmostEqual(wrap_0_pi(1.2 * pi), 0.8 * pi)
        self.assertAlmostEqual(wrap_0_pi(-0.2 * pi), 0.2 * pi)
        nt.assert_array_almost_equal(
            wrap_0_pi([0, pi, 1.2 * pi, -0.2 * pi]), [0, pi, 0.8 * pi, 0.2 * pi]
        )

        self.assertAlmostEqual(wrap_mpi2_pi2(0), 0)
        self.assertAlmostEqual(wrap_mpi2_pi2(-0.5 * pi), -0.5 * pi)
        self.assertAlmostEqual(wrap_mpi2_pi2(0.5 * pi), 0.5 * pi)
        self.assertAlmostEqual(wrap_mpi2_pi2(0.6 * pi), 0.4 * pi)
        self.assertAlmostEqual(wrap_mpi2_pi2(-0.6 * pi), -0.4 * pi)
        nt.assert_array_almost_equal(
            wrap_mpi2_pi2([0, -0.5 * pi, 0.5 * pi, 0.6 * pi, -0.6 * pi]),
            [0, -0.5 * pi, 0.5 * pi, 0.4 * pi, -0.4 * pi],
        )

        for angle_factor in (0, 0.3, 0.5, 0.8, 1.0, 1.3, 1.5, 1.7, 2):
            theta = angle_factor * pi
            self.assertAlmostEqual(angle_wrap(theta), wrap_mpi_pi(theta))
            self.assertAlmostEqual(angle_wrap(-theta), wrap_mpi_pi(-theta))
            self.assertAlmostEqual(angle_wrap(theta=theta, mode="-pi:pi"), wrap_mpi_pi(theta))
            self.assertAlmostEqual(angle_wrap(theta=-theta, mode="-pi:pi"), wrap_mpi_pi(-theta))
            self.assertAlmostEqual(angle_wrap(theta=theta, mode="0:2pi"), wrap_0_2pi(theta))
            self.assertAlmostEqual(angle_wrap(theta=-theta, mode="0:2pi"), wrap_0_2pi(-theta))
            self.assertAlmostEqual(angle_wrap(theta=theta, mode="0:pi"), wrap_0_pi(theta))
            self.assertAlmostEqual(angle_wrap(theta=-theta, mode="0:pi"), wrap_0_pi(-theta))
            self.assertAlmostEqual(angle_wrap(theta=theta, mode="-pi/2:pi/2"), wrap_mpi2_pi2(theta))
            self.assertAlmostEqual(angle_wrap(theta=-theta, mode="-pi/2:pi/2"), wrap_mpi2_pi2(-theta))
            with self.assertRaises(ValueError):
                angle_wrap(theta=theta, mode="foo")

    def test_angle_stats(self):
        theta = np.linspace(3 * pi / 2, 5 * pi / 2, 50)
        self.assertAlmostEqual(angle_mean(theta), 0)
        self.assertAlmostEqual(angle_std(theta), 0.9717284050981313)

        theta = np.linspace(pi / 2, 3 * pi / 2, 50)
        self.assertAlmostEqual(angle_mean(theta), pi)
        self.assertAlmostEqual(angle_std(theta), 0.9717284050981313)

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
