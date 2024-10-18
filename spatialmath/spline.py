# Copyright (c) 2024 Boston Dynamics AI Institute LLC.
# MIT Licence, see details in top-level file: LICENCE

"""
Classes for parameterizing a trajectory in SE3 with splines. 
"""

from abc import ABC, abstractmethod
from functools import cached_property
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import BSpline, CubicSpline
from scipy.spatial.transform import Rotation, RotationSpline

from spatialmath import SE3, SO3, Twist3
from spatialmath.base.transforms3d import tranimate


class SplineSE3(ABC):

    def __init__(self) -> None:
        self.control_poses: SE3

    @abstractmethod
    def __call__(self, t: float) -> SE3:
        pass

    def visualize(
        self,
        num_samples: int,
        pose_marker_length: float = 0.2,
        animate: bool = False,
        ax: Optional[plt.Axes]  = None,
        input_poses: Optional[List[SE3]] = None
    ) -> None:
        """Displays an animation of the trajectory with the control poses."""
        if ax is None:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(projection="3d")

        samples = [self(t) for t in np.linspace(0, 1, num_samples)]
        if not animate:
            x = [pose.x for pose in samples]
            y = [pose.y for pose in samples]
            z = [pose.z for pose in samples]
            ax.plot(x, y, z, "c", linewidth=1.0)  # plot spline fit

        x = [pose.x for pose in self.control_poses]
        y = [pose.y for pose in self.control_poses]
        z = [pose.z for pose in self.control_poses]
        ax.plot(x, y, z, "r*")  # plot control_poses

        if input_poses is not None:
            x = [pose.x for pose in input_poses]
            y = [pose.y for pose in input_poses]
            z = [pose.z for pose in input_poses]
            ax.plot(x, y, z, "go", fillstyle="none")  # plot compare to input poses

        if animate:
            tranimate(samples, repeat=True, length=pose_marker_length, wait=True)  # animate pose along trajectory
        else:
            plt.show()

class InterpSplineSE3(SplineSE3):
    """Class for an interpolated trajectory in SE3, as a function of time, through control_poses with a cubic spline.

    A combination of scipy.interpolate.CubicSpline and scipy.spatial.transform.RotationSpline (itself also cubic)
    under the hood.
    """
    _e = 1e-12
    def __init__(
        self,
        timepoints: List[float], 
        control_poses: List[SE3],
        *,
        normalize_time: bool = True,
        bc_type: str = "not-a-knot",  # not-a-knot is scipy default; None is invalid
    ) -> None:
        """Construct a InterpSplineSE3 object

        Extends the scipy CubicSpline object
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html#cubicspline

        Args :
            timepoints : list of times corresponding to provided poses
            control_poses : list of SE3 objects that govern the shape of the spline.
            normalize_time : flag to map times into the range [0, 1]
            bc_type : boundary condition provided to scipy CubicSpline backend.
                      string options: ["not-a-knot" (default), "clamped", "natural", "periodic"].
                      For tuple options and details see the scipy docs link above.
        """
        super().__init__()
        self.control_poses = control_poses
        self.timepoints = np.array(timepoints)

        if self.timepoints[-1] < self._e:
            raise ValueError("Difference between start and end timepoints is less than {self._e}")
        
        if len(self.control_poses) != len(self.timepoints):
            raise ValueError("Length of control_poses and timepoints must be equal.")
        
        if len(self.timepoints) < 2:
            raise ValueError("Need at least 2 data points to make a trajectory.")

        if normalize_time:
            self.timepoints = self.timepoints - self.timepoints[0]
            self.timepoints = self.timepoints / self.timepoints[-1]

        self.spline_xyz = CubicSpline(
            self.timepoints, 
            np.array([pose.t for pose in self.control_poses]), 
            bc_type=bc_type
        )
        self.spline_so3 = RotationSpline(self.timepoints, Rotation.from_matrix(np.array([(pose.R) for pose in self.control_poses])))

    def __call__(self, t: float) -> SE3:
        """Compute function value at t. 
        Return:
            pose: SE3
        """
        return SE3.Rt(t=self.spline_xyz(t), R=self.spline_so3(t).as_matrix())

    def derivative(self, t: float) -> Twist3:
        linear_vel = self.spline_xyz.derivative()(t)
        angular_vel = self.spline_so3(t, 1) #1 is angular rate, 2 is angular acceleration
        return Twist3(linear_vel, angular_vel)


class SplineFit:

    def __init__(
        self, 
        time_data: List[float], 
        pose_data: List[SE3],
    ) -> None:
        self.time_data = time_data
        self.pose_data = pose_data

        self.xyz_data = np.array([pose.t for pose in pose_data])
        self.so3_data = Rotation.from_matrix(np.array([(pose.R) for pose in pose_data]))

        self.spline: Optional[SplineSE3] = None

    def stochastic_downsample_interpolation(
        self, 
        epsilon_xyz: float = 1e-3, 
        epsilon_angle: float = 1e-1,
        normalize_time: bool = True,
        bc_type: str = "not-a-knot",
    ) -> Tuple[InterpSplineSE3, List[int]]:
        """
            Return:
                downsampled interpolating spline, 
                list of removed indices from input data
        """
        spline = InterpSplineSE3(
            self.time_data, 
            self.pose_data, 
            normalize_time = normalize_time, 
            bc_type=bc_type
        )
        chosen_indices: set[int] = set()
        interpolation_indices = list(range(len(self.pose_data)))
        interpolation_indices.remove(0)
        interpolation_indices.remove(len(self.pose_data) - 1)

        for _ in range(len(self.time_data) - 2):  # you must have at least 2 indices
            choices = list(set(interpolation_indices).difference(chosen_indices))

            index = np.random.choice(choices)

            chosen_indices.add(index)
            interpolation_indices.remove(index)

            spline.spline_xyz = CubicSpline(self.time_data[interpolation_indices], self.xyz_data[interpolation_indices])
            spline.spline_so3 = RotationSpline(
                self.time_data[interpolation_indices], self.so3_data[interpolation_indices]
            )

            time = self.time_data[index]
            angular_error = SO3(self.pose_data[index]).angdist(SO3(spline.spline_so3(time).as_matrix()))
            euclidean_error = np.linalg.norm(self.pose_data[index].t - spline.spline_xyz(time))
            if (angular_error > epsilon_angle) or (euclidean_error > epsilon_xyz):
                interpolation_indices.insert(int(np.searchsorted(interpolation_indices, index, side="right")), index)

        self.spline = spline
        return spline, interpolation_indices
    
    def max_angular_error(self) -> float:
        return np.max(self.angular_errors)

    @cached_property
    def angular_errors(self) -> List[float]:
        return [
            pose.angdist(self.spline(t))
            for pose, t in zip(self.pose_data, self.time_data)
        ]

    def max_euclidean_error(self) -> float:
        return np.max(self.euclidean_errors)

    @cached_property
    def euclidean_errors(self) -> List[float]:
        return [
            np.linalg.norm(pose.t - self.spline(t).t)
            for pose, t in zip(self.pose_data, self.time_data)
        ]


class BSplineSE3(SplineSE3):
    """A class to parameterize a trajectory in SE3 with a 6-dimensional B-spline.

    The SE3 control poses are converted to se3 twists (the lie algebra) and a B-spline
    is created for each dimension of the twist, using the corresponding element of the twists
    as the control point for the spline.

    For detailed information about B-splines, please see this wikipedia article.
    https://en.wikipedia.org/wiki/Non-uniform_rational_B-spline
    """

    def __init__(
        self,
        control_poses: List[SE3],
        degree: int = 3,
        knots: Optional[List[float]] = None,
    ) -> None:
        """Construct BSplineSE3 object. The default arguments generate a cubic B-spline
        with uniformly spaced knots.

        - control_poses: list of SE3 objects that govern the shape of the spline.
        - degree: int that controls degree of the polynomial that governs any given point on the spline.
        - knots: list of floats that govern which control points are active during evaluating the spline
        at a given t input. If none, they are automatically, uniformly generated based on number of control poses and
        degree of spline.
        """
        super().__init__()
        self.control_poses = control_poses

        # a matrix where each row is a control pose as a twist
        # (so each column is a vector of control points for that dim of the twist)
        self.control_pose_matrix = np.vstack(
            [np.array(element.twist()) for element in control_poses]
        )

        self.degree = degree

        if knots is None:
            knots = np.linspace(0, 1, len(control_poses) - degree + 1, endpoint=True)
            knots = np.append(
                [0.0] * degree, knots
            )  # ensures the curve starts on the first control pose
            knots = np.append(
                knots, [1] * degree
            )  # ensures the curve ends on the last control pose
        self.knots = knots

        self.splines = [
            BSpline(knots, self.control_pose_matrix[:, i], degree)
            for i in range(0, 6)  # twists are length 6
        ]

    def __call__(self, t: float) -> SE3:
        """Returns pose of spline at t.

        t: Normalized time value [0,1] to evaluate the spline at.
        """
        twist = np.hstack([spline(t) for spline in self.splines])
        return SE3.Exp(twist)


