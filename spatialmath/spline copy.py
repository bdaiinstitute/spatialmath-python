# Copyright (c) 2024 Boston Dynamics AI Institute LLC.
# MIT Licence, see details in top-level file: LICENCE

"""
Classes for parameterizing a trajectory in SE3 with B-splines. 

Copies parts of the API from scipy's B-spline class.
"""

from typing import Any, Dict, List, Optional
from scipy.interpolate import BSpline
from spatialmath import SE3, Twist3, SO3
import numpy as np
import matplotlib.pyplot as plt
from spatialmath.base.transforms3d import tranimate, trplot
from scipy.optimize import minimize
from scipy.interpolate import splrep


def weighted_average_SE3_metric(a: Twist3 | SE3, b: Twist3 | SE3, w: float = 0.5) -> float:
    """A positive definite distance metric for SE3.

    The metric is: (1-w)*translation_distance + w*angular_distance

    Note that there isn't a
    great "natural" choice for a metric on SE3 (see Appendix A.3.2 in
    'A Mathematical Introduction to Robotic Manipulation' by Murray, Li, Sastry)

    spatialmath has options for the rotational metrics from
    'Metrics for 3D Rotations: Comparison and Analysis' by Du Q. Huynh.
    This uses the 'best choice' from that paper.

    Args:
        a: a twist from SE3.twist, or an SE3 object
        b: a twist from SE3.twist, or an SE3 object
        w: a float that represents the relative weighting between the rotational and translation distances.

    Returns:
        a float for the 'distance' between two SE3 poses

    """
    if w> 1 or w < 0:
        raise ValueError(f"Weight w={w} is outside the range [0,1].")

    if isinstance(a, Twist3):
        a = SE3.Exp(a)
    
    if isinstance(b, Twist3):
        b = SE3.Exp(b)

    # if np.linalg.norm(a - b) < 1e-6:
    #     return 0.0
    
    angular_distance = a.angdist(b)
    translation_distance = np.linalg.norm(a.t - b.t)

    return (1 - w) * translation_distance + w * angular_distance


class TrajectoryTwist:

    def __init__(self, poses: List[SE3], timestamps: List[float]) -> None:
        self.poses = poses
        self.timestamps = timestamps
        self.twists = []
        
    def metric(self, target_pose: SE3, seed_twist: Twist3):
        return weighted_average_SE3_metric(target_pose, SE3.Exp(seed_twist))
    
    def fit(self):
        twists = []
        for pose in self.poses:
            if len(twists) == 0:
                twists.append(pose.twist())
            else:
               f = lambda x: self.metric(pose, x)
               twists.append(Twist3(minimize(f, twists[-1].A).x))

        self.twists = twists
    
    def visualize(self):
        fig = plt.figure(figsize=(40, 20))
        ax = fig.add_subplot(1, 2, 1, projection="3d")
        trplot(
            [np.array(self.poses)],
            ax=ax,
            length=0.2
        )  # plot control points
        # import ipdb; ipdb.set_trace()
        trplot(
            [np.array([twist.exp() for twist in self.twists])],
            ax=ax,
            length=0.2,
            color="cornflowerblue",
        )  # plot data points
        twist_data = np.vstack([twist.A for twist in self.twists])
        ax_2: plt.Axes = fig.add_subplot(1, 2, 2)
        ax_2.plot(
            self.timestamps,
            twist_data[:, 0],
            "o",
            self.timestamps,
            twist_data[:, 1],
            "o",
            self.timestamps,
            twist_data[:, 2],
            "o",
            self.timestamps,
            twist_data[:, 3],
            "o",
            self.timestamps,
            twist_data[:, 4],
            "o",
            self.timestamps,
            twist_data[:, 5],
            "o",
        )
        plt.show()
    

class CubicBSplineSE3:
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
    ) -> None:
        """Construct BSplineSE3 object. The default arguments generate a cubic B-spline
        with uniformly spaced knots.

        - control_poses: list of SE3 objects that govern the shape of the spline.
        - degree: int that controls degree of the polynomial that governs any given point on the spline.
        - knots: list of floats that govern which control points are active during evaluating the spline
        at a given t input. If none, they are automatically, uniformly generated based on number of control poses and
        degree of spline.
        """

        self.basis_coefficent_matrix = 1/6*np.array([
            [6, 0, 0, 0],
            [5, 3, -3, 1],
            [1, 3, 3, -2],
            [0, 0, 0, 1]
        ])
        self.n = len(control_poses)
        self.control_poses = control_poses
        self.degree = 3

        # use open uniform knot vector
        knots = np.linspace(0, 1, len(control_poses)-2, endpoint=True)
        knots = np.append(
            [0] * self.degree, knots
        )  # ensures the curve starts on the first control pose
        knots = np.append(
            knots, [1] * self.degree
        )  # ensures the curve ends on the last control pose
        self.knots = knots
        

    def __call__(self, time:float):   
        """Returns pose of spline at t.

        t: Normalized time value [0,1] to evaluate the spline at.
        """     
        current_pose = SE3()
        
        spline_no_coeff = BSpline.design_matrix([time], self.knots, self.degree)  #the B in sum(alpha_i * B_i) = S(t)

        rows,cols = spline_no_coeff.nonzero()
        for row,col in zip(rows,cols): 
            current_pose = self.control_poses[col].interp1(spline_no_coeff[row,col]) * current_pose

        
        return current_pose

    def visualize(
        self,
        num_samples: int,
        length: float = 0.2,
        repeat: bool = False,
        ax: Optional[plt.Axes] = None,
        kwargs_trplot: Dict[str, Any] = {"color": "green"},
        kwargs_tranimate: Dict[str, Any] = {"wait": True},
        kwargs_plot: Dict[str, Any] = {},
    ) -> None:
        """Displays an animation of the trajectory with the control poses."""
        out_poses = [self(t) for t in np.linspace(0, 1, num_samples)]
        x = [pose.x for pose in out_poses]
        y = [pose.y for pose in out_poses]
        z = [pose.z for pose in out_poses]

        if ax is None:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(projection="3d")

        trplot(
            [np.array(self.control_poses)], ax=ax, length=length, **kwargs_trplot
        )  # plot control points
        ax.plot(x, y, z, **kwargs_plot)  # plot x,y,z trajectory

        tranimate(
            out_poses, repeat=repeat, length=length, **kwargs_tranimate
        )  # animate pose along trajectory

        
class BSplineTwist3:
    """A class to parameterize a trajectory in SE3 with a 6-dimensional B-spline.

    The SE3 control poses are converted to se3 twists (the lie algebra) and a B-spline
    is created for each dimension of the twist, using the corresponding element of the twists
    as the control point for the spline.

    For detailed information about B-splines, please see this wikipedia article.
    https://en.wikipedia.org/wiki/Non-uniform_rational_B-spline
    """

    def __init__(
        self,
        control_twists: List[Twist3],
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

        self.control_twists = control_twists

        # a matrix where each row is a control pose as a twist
        # (so each column is a vector of control points for that dim of the twist)
        self.control_twist_matrix = np.vstack(
            [element for element in control_twists]
        )

        self.degree = degree

        if knots is None:
            knots = np.linspace(0, 1, len(control_twists) - degree + 1, endpoint=True)
            knots = np.append(
                [0.0] * degree, knots
            )  # ensures the curve starts on the first control pose
            knots = np.append(
                knots, [1] * degree
            )  # ensures the curve ends on the last control pose
        self.knots = knots

        self.splines = [
            BSpline(knots, self.control_twist_matrix[:, i], degree)
            for i in range(0, 6)  # twists are length 6
        ]

    def __call__(self, t: float) -> SE3:
        """Returns pose of spline at t.

        t: Normalized time value [0,1] to evaluate the spline at.
        """
        twist = np.hstack([spline(t) for spline in self.splines])
        return twist


class FitBSplineTwist3:

    def __init__(self, twist_data: np.ndarray, timestamps, knots) -> None:
        """twist data is nx6 matrix where a row is a twist"""
        
        self.twist_data = twist_data
        self.pose_data = [SE3.Exp(twist) for twist in twist_data]
        self.timestamps = timestamps
        self.spline = BSplineTwist3([Twist3() for i in range(0,8)])
        self.knots = knots
        self.degree = 3
        self.norm_ratio = 0.0

        t = np.array(self.timestamps)
        t = t - t[0]  # shift start to 0
        t = t / t[-1]  # normalize range to [0,1]

        self.normalized_timestamps = t


    def update_control_twists(self, control_twists: np.array):
        """control_twists is nx6 matrix where a row is a twist"""
        self.spline = BSplineTwist3([row for row in control_twists])

    def L1_over_euclidian_distance_error(self, control_twists: Optional[np.array] = None):
        """ control twists are input as a vector from a flattened nx6 matrix here
        """
        if control_twists is not None:
            control_twists = control_twists.reshape(int(len(control_twists)/6), 6)
            self.update_control_twists(control_twists)
        
        distances = np.array([np.linalg.norm(SE3.Exp(self.spline(timestamp)).t - pose.t) for timestamp, pose in zip(self.timestamps,self.pose_data, strict=True)])
        return np.linalg.norm(distances, ord=np.inf)
    
    def L1_over_SE3_metric_error(self, control_twists: Optional[np.array] = None):
        """ control twists are input as a vector from a flattened nx6 matrix here
        """
        if control_twists is not None:
            control_twists = control_twists.reshape(int(len(control_twists)/6), 6)
            self.update_control_twists(control_twists)
        
        distances = np.array([weighted_average_SE3_metric(SE3.Exp(self.spline(timestamp)), pose, self.norm_ratio) for timestamp, pose in zip(self.timestamps,self.pose_data, strict=True)])
        return np.linalg.norm(distances, ord=np.inf)

    def fit2(self):
        result = minimize(self.L1_over_SE3_metric_error, self.spline.control_twist_matrix.flatten())
        self.update_control_twists(result.x.reshape(int(len(result.x)/6), 6))

    def fit(self) -> None:
        """Fits a b spline to the input SE3 trajectory and outputs control poses.

        Assumes timestamps are monotonically increasing / are sorted.
        """
        # data_as_twists = np.vstack([np.array(pose.twist()) for pose in self.twist_data])
        # import ipdb; ipdb.set_trace()
        control_twists = np.vstack(
            [
                splrep(self.normalized_timestamps, self.twist_data[:, i], k=self.degree, t=self.knots)[1][
                    0 : len(self.knots) + self.degree + 1
                ]
                for i in range(6)
            ]
        ).T

        self.spline = BSplineTwist3([row for row in control_twists])

    def visualize(
        self,
        num_samples: int,
        length: float = 0.5,
        repeat: bool = True,
        kwargs_trplot: Dict[str, Any] = {"color": "green"},
        kwargs_tranimate: Dict[str, Any] = {"wait": True, "dims": [-5, 5, -5, 5, -5, 5]},
        kwargs_plot: Dict[str, Any] = {},
    ) -> None:
        """Displays an animation of the trajectory with the control poses."""

        fig = plt.figure(figsize=(40, 20))
        ax_1: plt.Axes = fig.add_subplot(1, 2, 1)

        times = np.linspace(0, 1, num_samples)

        fit_twists = np.vstack([self.spline(t) for t in times])
        # fit_twists = np.vstack([pose.UnitQuaternion().A for pose in fit_poses])
        ax_1.plot(
            times,
            fit_twists[:, 0],
            times,
            fit_twists[:, 1],
            times,
            fit_twists[:, 2],
            times,
            fit_twists[:, 3],
            times,
            fit_twists[:, 4],
            times,
            fit_twists[:, 5],
        )

        # pose_data_twists = np.vstack([pose.twist().A for pose in self.twist_data])
        ax_1.plot(
            self.timestamps,
            self.twist_data[:, 0],
            "o",
            self.timestamps,
            self.twist_data[:, 1],
            "o",
            self.timestamps,
            self.twist_data[:, 2],
            "o",
            self.timestamps,
            self.twist_data[:, 3],
            "o",
            self.timestamps,
            self.twist_data[:, 4],
            "o",
            self.timestamps,
            self.twist_data[:, 5],
            "o",
        )
        ax_1.plot(
            self.spline.knots[self.degree-1:-self.degree+1],
            self.spline.control_twist_matrix[:, 0],
            "x",
            self.spline.knots[self.degree-1:-self.degree+1],
            self.spline.control_twist_matrix[:, 1],
            "x",
            self.spline.knots[self.degree-1:-self.degree+1],
            self.spline.control_twist_matrix[:, 2],
            "x",
            self.spline.knots[self.degree-1:-self.degree+1],
            self.spline.control_twist_matrix[:, 3],
            "x",
            self.spline.knots[self.degree-1:-self.degree+1],
            self.spline.control_twist_matrix[:, 4],
            "x",
            self.spline.knots[self.degree-1:-self.degree+1],
            self.spline.control_twist_matrix[:, 5],
            "x",
        )
        ax_1.legend(
            labels=[
                "x_fit",
                "y_fit",
                "z_fit",
                "x_rot_fit",
                "y_rot_fit",
                "z_rot_fit",
                "x_true",
                "y_true",
                "z_true",
                "x_rot_true",
                "y_rot_true",
                "z_rot_true",
                "x_control",
                "y_control",
                "z_control",
                "x_rot_control",
                "y_rot_control",
                "z_rot_control",
            ]
        )

        ax_2: plt.Axes = fig.add_subplot(1, 2, 2, projection="3d")
        # ax_2.plot(x, y, z, **kwargs_plot)  # plot x,y,z trajectory
        # import ipdb; ipdb.set_trace()
        trplot(
            [np.array([SE3.Exp(twist) for twist in self.spline.control_twists])],
            ax=ax_2,
            length=length,
            **kwargs_trplot,
        )  # plot control points
        trplot(
            [np.array([SE3.Exp(twist) for twist in self.twist_data])],
            ax=ax_2,
            length=0.25 * length,
            color="cornflowerblue",
        )  # plot data points

        fit_poses = [SE3.Exp(twist) for twist in fit_twists]
        # for pose in fit_poses:
        #     print(pose.ishom())
        tranimate(fit_poses, repeat=repeat, length=length, **kwargs_tranimate)  # animate pose along trajectory


def example_bspline_fit_scipy():

    num_data_points = 9
    num_samples = 100
    frequency = 2
    scale = 2

    timestamps = np.linspace(0, 1, num_data_points)
    x = np.array([t for t in timestamps])
    trajectory = [
        SE3.Rt(t = [e, np.sin(e * 2*np.pi* frequency), np.sin(e * 2*np.pi * frequency)], R= SO3.Rx( e*2*np.pi* frequency))
        for e in x
    ]
    print("Control points")
    for pose in trajectory:
        print(pose)

    se3_spline = CubicBSplineSE3(control_poses=trajectory)
    # print("Spline")
    # for knot in np.unique(se3_spline.knots):
    #     print(se3_spline.eval_spline(knot-0.001))
    #     print(se3_spline.eval_spline(knot+0.001))

    # import ipdb; ipdb.set_trace()
    se3_spline.visualize(num_samples=100, repeat=True)

    # import ipdb; ipdb.set_trace()
    # trajectory = [
    #     SE3.Rt(
    #         t=[scale * t, scale * np.cos(t * frequency * np.pi), scale * np.cos(t * frequency * np.pi)],
    #         R=SO3.Ry(t * 1.5 * frequency * np.pi),
    #     )
    #     for t in timestamps
    # ]
    # delta_trajectories = np.vstack([trajectory[i+1].delta(trajectory[i]) for i in range(0,len(trajectory)-1)])
    
    # trajectory = np.vstack(
    #     [
    #         Twist3(
    #             [
    #                 scale * t,
    #                 scale * np.cos(t * frequency * np.pi),
    #                 scale * np.sin(t * frequency * np.pi),
    #                 t * 1.5 * frequency * np.pi,
    #                 1,
    #                 2,
    #             ]
    #         ).A
    #         for t in timestamps
    #     ]
    # )

    # num_knots = 4
    # knots = np.linspace(0, 1, num_knots + 2)[1:-1]
    # print(f"knots: {knots}")

    # fit = FitBSplineTwist3(twist_data=delta_trajectories, timestamps=timestamps[0:-1], knots=knots)
    # fit.pose_data = trajectory
    # fit.fit()
    # print(fit.L1_euclidian_distance_error())
    # fit.visualize(num_samples=num_samples)
    # fit.norm_ratio = 0.0
    # fit.fit2()
    # fit.norm_ratio = 0.5
    # fit.fit2()
    # print(fit.L1_over_euclidian_distance_error())

    # fit.visualize(num_samples=num_samples, repeat=True)

    # traj_class = TrajectoryTwist(poses=trajectory, timestamps=timestamps)    
    # traj_class.fit()
    # import ipdb; ipdb.set_trace()
    # traj_class.visualize()


if __name__ == "__main__":
    example_bspline_fit_scipy()
