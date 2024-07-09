import numpy.testing as nt
import numpy as np
import matplotlib.pyplot as plt
import unittest
import sys
import pytest

from spatialmath import BSplineSE3, SE3


class TestBSplineSE3(unittest.TestCase):
    control_poses = [
        SE3.Trans([e, 2 * np.cos(e / 2 * np.pi), 2 * np.sin(e / 2 * np.pi)])
        * SE3.Ry(e / 8 * np.pi)
        for e in range(0, 8)
    ]

    @classmethod
    def tearDownClass(cls):
        plt.close("all")

    def test_constructor(self):
        BSplineSE3(self.control_poses)

    def test_evaluation(self):
        spline = BSplineSE3(self.control_poses)
        nt.assert_almost_equal(spline(0).A, self.control_poses[0].A)
        nt.assert_almost_equal(spline(1).A, self.control_poses[-1].A)

    def test_visualize(self):
        spline = BSplineSE3(self.control_poses)
        spline.visualize(num_samples=100, repeat=False)
