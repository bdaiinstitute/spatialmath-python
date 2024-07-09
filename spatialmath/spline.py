# Copyright (c) 2024 Boston Dynamics AI Institute LLC.
# MIT Licence, see details in top-level file: LICENCE

"""
Classes for parameterizing a trajectory in SE3 with B-splines. 

Copies parts of the API from scipy's B-spline class.
"""

from typing import Any, Optional
from scipy.interpolate import BSpline
from spatialmath import SE3
import numpy as np
import matplotlib.pyplot as plt
from spatialmath.base.transforms3d import tranimate, trplot

class BSplineSE3:
    """ A class to parameterize a trajectory in SE3 with a 6-dimensional B-spline.

    The SE3 control poses are converted to se3 twists (the lie algebra) and a B-spline 
    is created for each dimension of the twist, using the corresponding element of the twists 
    as the control point for the spline.  

    For detailed information about B-splines, please see this wikipedia article. 
    https://en.wikipedia.org/wiki/Non-uniform_rational_B-spline
    """

    def __init__(self, control_poses: list[SE3], degree: int = 3, knots: Optional[list[float]] = None) -> None:
        """ Construct BSplineSE3 object. The default arguments generate a cubic B-spline
        with uniformly spaced knots.

        - control_poses: list of SE3 objects that govern the shape of the spline. 
        - degree: int that controls degree of the polynomial that governs any given point on the spline.
        - knots: list of floats that govern which control points are active during evaluating the spline 
        at a given t input.
        """
        
        self.control_poses = control_poses
        
        # a matrix where each row is a control pose as a twist
        # (so each column is a vector of control points for that dim of the twist)
        self.control_pose_matrix = np.vstack([np.array(element.twist()) for element in control_poses]) 
        
        self.degree = degree 
        
        if knots is None:
            knots=np.linspace(0,1,len(control_poses)-2,endpoint=True)
            knots=np.append([0,0,0],knots) # ensures the curve starts on the first control pose
            knots=np.append(knots,[1,1,1])  # ensures the curve ends on the last control pose
        self.knots = knots  

        self.splines = [ BSpline(knots, self.control_pose_matrix[:, i], degree) for i in range(0, 6) ]

    def __call__(self, t: float) -> SE3:
        """ Returns pose of spline at t.

        t: Normalized time value [0,1] to evaluate the spline at.
        """
        twist = np.hstack([spline(t) for spline in self.splines])
        return SE3.Exp(twist)
    
    def visualize(self, num_samples: int, repeat: bool = False) -> None:
        """ Displays an animation of the trajectory with the control poses.
        """
        out_poses = [self(i) for i in np.linspace(0,1,num_samples)]
        x = [pose.x for pose in out_poses]
        y = [pose.y for pose in out_poses]
        z = [pose.z for pose in out_poses]
        
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(projection='3d')
        trplot([np.array(self.control_poses)],ax=ax, length= 1.0, color="green") # plot control points
        ax.plot(x,y,z) # plot x,y,z trajectory

        tranimate(out_poses, repeat=repeat, wait=True, length = 1.0) # animate pose along trajectory

def main():
    degree = 3
    control_poses = [
        SE3.Trans(
            [e, 2*np.cos(e/2 * np.pi), 2*np.sin(e/2 * np.pi)]
        )
        *SE3.Ry(e/8 * np.pi) for e in range(0,8)
    ]
    
    spline = BSplineSE3(control_poses, degree)
    spline.visualize(num_samples=100, repeat=True)
    
    

if __name__ == "__main__":
    main()