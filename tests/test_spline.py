import numpy.testing as nt
import numpy as np
import matplotlib.pyplot as plt
import unittest
from spatialmath import FitCubicBSplineSE3, CubicBSplineSE3, SE3, SO3


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
        CubicBSplineSE3(self.control_poses)

    def test_evaluation(self):
        spline = CubicBSplineSE3(self.control_poses)
        nt.assert_almost_equal(spline(0).A, self.control_poses[0].A)
        nt.assert_almost_equal(spline(1).A, self.control_poses[-1].A)

    def test_visualize(self):
        spline = CubicBSplineSE3(self.control_poses)
        spline.visualize(num_samples=100, repeat=False)

class TestFitBSplineSE3(unittest.TestCase):

    num_data_points = 16
    num_samples = 100
    num_control_points = 6

    timestamps = np.linspace(0, 1, num_data_points)
    trajectory = [
        SE3.Rt(t = [t*4, 4*np.sin(t * 2*np.pi* 0.5), 4*np.cos(t * 2*np.pi * 0.5)], 
               R= SO3.Rx( t*2*np.pi* 0.5))
        for t in timestamps
    ]

    @classmethod
    def tearDownClass(cls):
        plt.close("all")

    def test_constructor(self):
        pass

    def test_evaluation_and_visualization(self):
        fit_se3_spline = FitCubicBSplineSE3(self.trajectory, self.timestamps, num_control_points=self.num_control_points)

        result = fit_se3_spline.fit(disp=True)

        assert len(result) == 2
        assert fit_se3_spline.objective_function_xyz() < 0.01
        assert fit_se3_spline.objective_function_so3() < 0.01

        fit_se3_spline.visualize(num_samples=self.num_samples, repeat=False, length=0.4, kwargs_tranimate={"wait": True, "interval" : 10})