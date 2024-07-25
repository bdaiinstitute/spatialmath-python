# Copyright (c) 2024 Boston Dynamics AI Institute LLC.
# MIT Licence, see details in top-level file: LICENCE

"""
Classes for parameterizing a trajectory in SE3 with B-splines. 

Copies parts of the API from scipy's B-spline class.
"""

import time
from typing import Any, Dict, List, Optional, Tuple
from scipy.interpolate import BSpline, CubicSpline
from scipy.spatial.transform import RotationSpline, Rotation
from spatialmath import SE3, Twist3, SO3
import numpy as np
import matplotlib.pyplot as plt
from spatialmath.base.transforms3d import tranimate, trplot
from scipy.optimize import minimize
from scipy.interpolate import splrep


class CubicBSplineSE3:
    """A class to parameterize a trajectory in SE3 with a cubic B-spline.

    The position and orientation are calculated disjointly. The position uses the
    "classic" B-spline formulation. The orientation calculation is based on 
    interpolation between the identity element and the control SO3 element, them applying 
    each interpolated SO3 element as a group action. 

    For detailed information about B-splines, please see this wikipedia article.
    https://en.wikipedia.org/wiki/Non-uniform_rational_B-spline
    """

    degree = 3
    def __init__(
        self,
        control_poses: List[SE3],
        time_range: Tuple[float,float] = (0.0, 1.0)
    ) -> None:
        """Construct a CubicBSplineSE3 object with a open uniform knot vector.

        - control_poses: list of SE3 objects that govern the shape of the spline.
        """

        self.control_poses = control_poses
        self.knots = self.knots_from_num_control_poses(len(control_poses))
        self.time_range = time_range
        
    def __call__(self, time:float):   
        """Returns pose of spline at t.

        t: Time value inside the time range to evaluate the spline at.
        """     
           
        time = self.normalize_time(time)

        spline_no_coeff = BSpline.design_matrix([time], self.knots, self.degree)  #the B in sum(alpha_i * B_i) = S(t)
        rows,cols = spline_no_coeff.nonzero()

        current_so3 = SO3()
        for row,col in zip(rows,cols): 
            control_so3: SO3 = SO3(self.control_poses[col]).norm()
            try:
                current_so3 = control_so3.interp1(spline_no_coeff[row,col]) * current_so3
            except Exception:
                import ipdb; ipdb.set_trace()

        xyz = np.array([0,0,0])
        for row,col in zip(rows,cols): 
            control_point = self.control_poses[col].t
            xyz = xyz + control_point*spline_no_coeff[row,col]

        return SE3.Rt(t = xyz, R = current_so3).norm()
    
    def normalize_time(self, time: float) -> float:
        return (time - self.time_range[0]) / (self.time_range[1] - self.time_range[0])

    @classmethod
    def knots_from_num_control_poses(self, num_control_poses: int):
        """ Return open uniform knots vector based on number of control poses. """
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
            out_poses, repeat=repeat, length=length, **{"wait": True}
        )  # animate pose along trajectory


class CubicSplineSE3:
    """  A combiation of scipy.interpolate.CubicSpline and 
    scipy.spatial.transform.RotationSpline (itself also cubic)
    """
    def __init__(
        self,
        timestamps: List[float],
        pose_data: List[SE3],
    ) -> None:
        """Construct a CubicSplineSE3 object

        - control_poses: list of SE3 objects that govern the shape of the spline.
        """

        self.pose_data = pose_data
        self.timestamps = np.array(timestamps)
        self.timestamps = self.timestamps - self.timestamps[0]
        self.timestamps = self.timestamps / self.timestamps[-1]
        
        self.xyz_data = np.array([pose.t for pose in pose_data]).copy()
        self.so3_data = Rotation.from_matrix(np.array([(pose.R) for pose in pose_data]))
        
        self.spline_xyz = CubicSpline(self.timestamps, self.xyz_data)
        self.spline_so3 = RotationSpline(self.timestamps, self.so3_data)

        self.available_indices = list(range(0, len(pose_data)))

        # import ipdb; ipdb.set_trace()


    def __call__(self, t: float, as_se3: bool = False) -> Any:
        if as_se3:
            return (SE3.Rt(t = self.spline_xyz(t), R = self.spline_so3(t).as_matrix() ))
        return (self.spline_xyz(t), self.spline_so3(t))
    

    def max_ang_error(self):
        """ Returns vector of angular distance between spline and data points.
        """
        return np.max([pose.angdist(SO3(self.spline_so3(timestamp).as_matrix())) for pose, timestamp in zip(self.pose_data, self.timestamps)])

    def max_distance_error(self):
        """ Returns vector of euclidean distance between spline and data points.
        """
        return np.max([np.linalg.norm(pose.t - self.spline_xyz(timestamp)) for pose, timestamp in zip(self.pose_data, self.timestamps)])


    def downsample(self, epsilon_xyz: float = 1e-3, epsilon_angle: float = 1e-1) -> int:
        chosen_indices = set()
        available_indices = self.available_indices.copy()
        available_indices.remove(0)
        available_indices.remove(len(self.pose_data)-1)

        for j in range(0, len(self.timestamps) - 4):
            choices = list(set(available_indices).difference(chosen_indices))
            index = np.random.choice(choices)
            chosen_indices.add(index)
            available_indices.remove(index)
            try:
                self.spline_xyz = CubicSpline(self.timestamps[available_indices], self.xyz_data[available_indices])
                self.spline_so3 = RotationSpline(self.timestamps[available_indices], self.so3_data[available_indices])
            except Exception as e :
                print(e)
                import ipdb; ipdb.set_trace()

            time = self.timestamps[index]
            ang_distance = self.pose_data[index].angdist(SO3(self.spline_so3(time).as_matrix()))
            euclidean_distance = np.linalg.norm(self.pose_data[index].t - self.spline_xyz(time))

            if ang_distance > epsilon_angle or euclidean_distance > epsilon_xyz:
                i = np.searchsorted(available_indices, index, side="right")
                available_indices.insert(i, index)

            # import ipdb; ipdb.set_trace()
        self.available_indices = available_indices
        return len(self.pose_data) - len(available_indices)
        
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
        out_poses = [self(t, True) for t in np.linspace(0, 1, num_samples)]
        x = [pose.x for pose in out_poses]
        y = [pose.y for pose in out_poses]
        z = [pose.z for pose in out_poses]

        if ax is None:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(projection="3d")

        # trplot(
        #     [np.array(self.control_poses)], ax=ax, length=length, **kwargs_trplot
        # )  # plot control points
        ax.plot(x, y, z)  # plot x,y,z trajectory

        x = [pose.x for pose in self.pose_data]
        y = [pose.y for pose in self.pose_data]
        z = [pose.z for pose in self.pose_data]
        ax.plot(x, y, z)  # plot x,y,z trajectory

        tranimate(
            out_poses, repeat=repeat, length=length, **kwargs_tranimate
        )  # animate pose along trajectory

class FitCubicBSplineSE3:

    def __init__(self, timestamps, pose_data: List[SE3], num_control_points: int = 6, method = "slsqp") -> None:
        """ 
            Timestamps should be sorted.
            Outputs control poses, but the optimization is done on the [pos,axis-angle], flattened into a 1d vector.
        """
        self.pose_data = pose_data
        self.timestamps = timestamps
        self.num_control_pose = num_control_points
        self.method = method

        # initialize control poses with data points
        sample_points = np.linspace(self.timestamps[0], self.timestamps[-1], self.num_control_pose)
        closest_timestamp_indices = np.searchsorted(self.timestamps, sample_points)
        for i, index in enumerate(closest_timestamp_indices):
            if index < 0:
                closest_timestamp_indices[i] = 0
            if index >= len(timestamps):
                closest_timestamp_indices[i] = len(timestamps) - 1
    
        self.spline = CubicBSplineSE3(
            control_poses=[SE3.CopyFrom(self.pose_data[index]) for index in closest_timestamp_indices],
            time_range=(self.timestamps[0], self.timestamps[-1])
        )

    def objective_function_xyz(self, xyz_flat: Optional[np.ndarray] = None):
        """L-infinity norm of euclidean distance between data points and spline"""

        # data massage
        if xyz_flat is not None:
            self._assign_xyz_to_control_poses(xyz_flat)

        # objective
        error_vector = self.euclidean_distance()
        return np.linalg.norm(error_vector, ord=np.inf)

    def objective_function_so3(self, so3_twists_flat: Optional[np.ndarray] = None):
        """L-infinity norm of angular distance between data points and spline"""

        # data massage
        if so3_twists_flat is not None:
            self._assign_so3_twists_to_control_poses(so3_twists_flat)

        # objective
        error_vector = self.ang_distance()
        return np.linalg.norm(error_vector, ord=np.inf)
    
    def fit(self, disp: bool = False):
        """ Find the spline control points that minimize the distance from the spline to the data points.
        """
        tic = time.time()
        so3_result = self.fit_so3(disp)
        toc = time.time()
        print(f"fit_so3 time: {toc - tic}")

        tic = time.time()
        pos_result = self.fit_xyz(disp)
        toc = time.time()
        print(f"fit_xyz time: {toc - tic}")

        return pos_result, so3_result

    def fit_xyz(self, disp: bool = False):
        """ Solve fitting problem for x,y,z coordinates.
        """
        xyz_flat = self._flatten_control_xyz()
        result = minimize(self.objective_function_xyz, xyz_flat, method=self.method, options = {"disp":disp})   
        self._assign_xyz_to_control_poses(result.x)

        return result
    
    def fit_so3(self, disp: bool = False):
        """ Solve fitting problem for SO3 coordinates.
        """
        so3_twists_flat = self._flatten_control_so3_twists()
        result = minimize(self.objective_function_so3, so3_twists_flat, method = self.method, options = {"disp":disp})    
        self._assign_so3_twists_to_control_poses(result.x)

        return result

    def _assign_xyz_to_control_poses(self, xyz_flat: np.ndarray) -> None:
        xyz_mat = xyz_flat.reshape(self.num_control_pose, 3) 
        for i, xyz in enumerate(xyz_mat):
            self.spline.control_poses[i].t = xyz

    def _assign_so3_twists_to_control_poses(self, so3_twists_flat: np.ndarray) -> None:
        so3_twists_mat = so3_twists_flat.reshape(self.num_control_pose, 3) 
        for i, so3_twist in enumerate(so3_twists_mat):
            self.spline.control_poses[i].R = SO3.Exp(so3_twist)

    def _flatten_control_xyz(self):
        return np.concatenate([pose.t for pose in self.spline.control_poses])

    def _flatten_control_so3_twists(self):
        return np.concatenate([SO3(pose.R).log(twist=True) for pose in self.spline.control_poses])

    def ang_distance(self):
        """ Returns vector of angular distance between spline and data points.
        """
        return [pose.angdist(self.spline(timestamp)) for pose, timestamp in zip(self.pose_data, self.timestamps)]

    def euclidean_distance(self):
        """ Returns vector of euclidean distance between spline and data points.
        """
        return [np.linalg.norm(pose.t - self.spline(timestamp).t) for pose, timestamp in zip(self.pose_data, self.timestamps)]

    def performance_profile(self):
        
        def tic_toc(f) -> float:
            tic = time.time()
            f()
            toc = time.time()
            duration = toc - tic
            print(f"{f.__name__} took {duration} seconds.")
            return (duration)
        
        tic_toc(self.ang_distance)
        tic_toc(self.euclidean_distance)
        tic_toc(self.objective_function_so3)
        tic_toc(self.objective_function_xyz)
        tic_toc(self._flatten_control_xyz)
        tic_toc(self._flatten_control_so3_twists)

        so3 = self._flatten_control_so3_twists()
        tic_toc(lambda : self._assign_so3_twists_to_control_poses(so3))

        xyz = self._flatten_control_xyz()
        tic_toc(lambda : self._assign_xyz_to_control_poses(xyz))
        
    def append_data(self, pose_data: List[SE3], timestamps) -> None:
        self.pose_data = [*self.pose_data, *pose_data]
        if self.timestamps[-1] > timestamps[0]:
            raise ValueError("New timestamps should be greater than old stamps.")
        self.timestamps = [*self.timestamps, *timestamps]
        self.spline.time_range = (self.timestamps[0], self.timestamps[-1])

    def visualize(
        self,
        num_samples: int,
        length: float = 0.2,
        repeat: bool = False,
        kwargs_trplot: Dict[str, Any] = {"color": "green"},
        kwargs_tranimate: Dict[str, Any] = {"wait": True},
        kwargs_plot: Dict[str, Any] = {},
    ) -> None:
        """Displays an animation of the trajectory with the control poses and data points."""
        out_poses = [self.spline(t) for t in np.linspace(0, 1, num_samples)]
        x = [pose.x for pose in out_poses]
        y = [pose.y for pose in out_poses]
        z = [pose.z for pose in out_poses]

        
        fig = plt.figure(figsize=(10, 10))
        
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(self.timestamps, self.ang_distance(), "--x")
        ax.plot(self.timestamps, self.euclidean_distance(), "--x")
        ax.legend(["angular distance", "euclidiean distance"])
        ax.set_xlabel("Time")

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
