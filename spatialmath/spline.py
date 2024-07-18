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


def weighted_average_SE3_metric(a: SE3, b: SE3, w: float = 0.5) -> float:
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

    # if np.linalg.norm(a - b) < 1e-6:
    #     return 0.0
    
    angular_distance = a.angdist(b)
    translation_distance = np.linalg.norm(a.t - b.t)

    return (1 - w) * translation_distance + w * angular_distance

class CubicBSplineSE3:
    """A class to parameterize a trajectory in SE3 with a 6-dimensional B-spline.

    The SE3 control poses are converted to se3 twists (the lie algebra) and a B-spline
    is created for each dimension of the twist, using the corresponding element of the twists
    as the control point for the spline.

    For detailed information about B-splines, please see this wikipedia article.
    https://en.wikipedia.org/wiki/Non-uniform_rational_B-spline
    """

    degree = 3
    def __init__(
        self,
        control_poses: List[SE3],
    ) -> None:
        """Construct BSplineSE3 object. The default arguments generate a cubic B-spline
        with a open uniform knot vector.

        - control_poses: list of SE3 objects that govern the shape of the spline.
        """

        self.control_poses = control_poses
        self.knots = self.knots_from_num_control_poses(len(control_poses))
        
    def __call__(self, time:float):   
        """Returns pose of spline at t.

        t: Normalized time value [0,1] to evaluate the spline at.
        """     
        
        
        spline_no_coeff = BSpline.design_matrix([time], self.knots, self.degree)  #the B in sum(alpha_i * B_i) = S(t)
        rows,cols = spline_no_coeff.nonzero()

        current_so3 = SO3()
        for row,col in zip(rows,cols): 
            control_so3: SO3 = SO3(self.control_poses[col])
            current_so3 = control_so3.interp1(spline_no_coeff[row,col]) * current_so3

        xyz = np.array([0,0,0])
        for row,col in zip(rows,cols): 
            control_point = self.control_poses[col].t
            xyz = xyz + control_point*spline_no_coeff[row,col]

        return SE3.Rt(t = xyz, R = current_so3)

    @classmethod
    def knots_from_num_control_poses(self, num_control_poses: int):
        # use open uniform knot vector
        knots = np.linspace(0, 1, num_control_poses-2, endpoint=True)
        knots = np.append(
            [0] * CubicBSplineSE3.degree, knots
        )  # ensures the curve starts on the first control pose
        knots = np.append(
            knots, [1] * CubicBSplineSE3.degree
        )  # ensures the curve ends on the last control pose
        return knots 

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

        
class FitCubicBSplineSE3:

    def __init__(self, pose_data: List[SE3], timestamps, num_control_points: int =6) -> None:
        """ 
            Timestamps should be normalized to [0,1] and sorted.
            Outputs control poses, but the optimization is done on the [pos,axis-angle], flattened into a 1d vector.
        """
        self.pose_data = pose_data
        self.timestamps = timestamps
        self.num_control_pose = num_control_points
        self.se3_rep_size = 6 #[x, y, z, axis-angle vec]

        # 
        sample_points = np.linspace(self.timestamps[0], self.timestamps[-1], self.num_control_pose)
        closest_timestamp_indices = np.searchsorted(self.timestamps, sample_points)
        for i, index in enumerate(closest_timestamp_indices):
            if index < 0:
                closest_timestamp_indices[i] = 0
            if index >= len(timestamps):
                closest_timestamp_indices[i] = len(timestamps) - 1
        
        # 
        self.spline = CubicBSplineSE3(control_poses=[SE3.CopyFrom(self.pose_data[index]) for index in closest_timestamp_indices])

    @classmethod
    def make_SE3_pose(self, row: np.ndarray):
        t = row[0:3]
        so3_twist = row[3:6]
 
        return SE3.Rt(t = t, R = SO3.Exp(so3_twist))
        
    @classmethod
    def make_SE3_rep(self, pose: SE3):
        so3_twist = SO3(pose.R).log(twist=True)
        return np.concatenate([pose.t, so3_twist])

    def objective_function_xyz(self, pos: np.ndarray):
        "L1 norm of SE3 distances between data points and spline"

        # data massage
        pos_matrix = pos.reshape(self.num_control_pose, 3)
        spline = CubicBSplineSE3(control_poses=[SE3.Trans(row) for row in pos_matrix])

        # objective
        error_vector = [ weighted_average_SE3_metric(spline(t), pose, w = 0.0) for t,pose in zip(self.timestamps, self.pose_data, strict=True) ]
        
        return np.linalg.norm(error_vector, ord=np.inf)

    def objective_function_so3(self, so3_twists_flat: np.ndarray):
        "L1 norm of SE3 distances between data points and spline"

        # data massage
        so3_twists_matrix = so3_twists_flat.reshape(self.num_control_pose, 3)
        spline = CubicBSplineSE3(control_poses=[SE3(SO3.Exp(row)) for row in so3_twists_matrix])

        # objective
        error_vector = [ weighted_average_SE3_metric(spline(t), pose, w = 1.0) for t,pose in zip(self.timestamps, self.pose_data, strict=True) ]
        
        return np.linalg.norm(error_vector, ord=np.inf)
    
    def objective_function_pose(self, control_pose_rep: np.ndarray):
        "L1 norm of SE3 distances between data points and spline"

        # data massage
        control_pose_matrix = control_pose_rep.reshape(self.num_control_pose, self.se3_rep_size)
        spline = CubicBSplineSE3(control_poses=[self.make_SE3_pose(row) for row in control_pose_matrix])

        # objective
        error_vector = [ weighted_average_SE3_metric(spline(t), pose) for t,pose in zip(self.timestamps, self.pose_data, strict=True) ]
        
        return np.linalg.norm(error_vector, ord=np.inf)

    def fit(self, disp: bool = False):

        so3_result = self.fit_so3(disp)
        pos_result = self.fit_xyz(disp)
        
        return pos_result, so3_result

    def fit_xyz(self, disp: bool = False):
        
        xyz_flat = np.concatenate([pose.t for pose in self.spline.control_poses])
        result = minimize(self.objective_function_xyz, xyz_flat, method="slsqp", options = {"disp":disp})   
        xyz_mat = result.x.reshape(self.num_control_pose, 3) 
        for i, xyz in enumerate(xyz_mat):
            self.spline.control_poses[i].t = xyz

        return result
    
    def fit_so3(self, disp: bool = False):
        so3_twists_flat = np.concatenate([SO3(pose.R).log(twist=True)for pose in self.spline.control_poses])
        result = minimize(self.objective_function_so3, so3_twists_flat, method="slsqp", options = {"disp":disp})    
        so3_twists_mat = result.x.reshape(self.num_control_pose, 3) 
        for i, so3_twist in enumerate(so3_twists_mat):
            self.spline.control_poses[i].R = SO3.Exp(so3_twist)
        return result

    def ang_distance(self):
        return [pose.angdist(self.spline(timestamp)) for pose, timestamp in zip(self.pose_data, self.timestamps)]

    def euclidean_distance(self):
        return [np.linalg.norm(pose.t - self.spline(timestamp).t) for pose, timestamp in zip(self.pose_data, self.timestamps)]

    def visualize(
        self,
        num_samples: int,
        length: float = 0.2,
        repeat: bool = False,
        kwargs_trplot: Dict[str, Any] = {"color": "green"},
        kwargs_tranimate: Dict[str, Any] = {"wait": True},
        kwargs_plot: Dict[str, Any] = {},
    ) -> None:
        """Displays an animation of the trajectory with the control poses."""
        out_poses = [self.spline(t) for t in np.linspace(0, 1, num_samples)]
        x = [pose.x for pose in out_poses]
        y = [pose.y for pose in out_poses]
        z = [pose.z for pose in out_poses]

        
        fig = plt.figure(figsize=(10, 10))
        
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(self.timestamps, self.ang_distance(), "--x")
        ax.plot(self.timestamps, self.euclidean_distance(), "--x")
        ax.legend(["angular distance", "euclidiean distance"])
        ax.set_xlabel("Normalized time")

        ax = fig.add_subplot(1, 2, 2, projection="3d")

        trplot(
            [np.array(self.spline.control_poses)], ax=ax, length=length*1.2, color= "green"
        )  # plot control points
        ax.plot(x, y, z, **kwargs_plot)  # plot x,y,z trajectory
        trplot(
            [np.array(self.pose_data)], ax=ax, length=length, color= "cornflowerblue", 
        )  # plot data points

        tranimate(
            out_poses, repeat=repeat, length=length, **kwargs_tranimate
        )  # animate pose along trajectory


def example_bspline_fit():

    num_data_points = 16
    num_samples = 100
    frequency = 0.5
    scale = 4

    timestamps = np.linspace(0, 1, num_data_points)
    trajectory = [
        SE3.Rt(t = [t*scale, scale*np.sin(t * 2*np.pi* frequency), scale*np.cos(t * 2*np.pi * frequency)], 
               R= SO3.Rx( t*2*np.pi* frequency))
        for t in timestamps
    ]

    fit_se3_spline = FitCubicBSplineSE3(trajectory, timestamps, num_control_points=6)
    
    result = fit_se3_spline.fit(disp=True)
    fit_se3_spline.visualize(num_samples=num_samples, repeat=True, length=0.4, kwargs_tranimate={"wait": True, "interval" : 400})


if __name__ == "__main__":
    example_bspline_fit()
