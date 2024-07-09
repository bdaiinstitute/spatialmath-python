# Copyright (c) 2024 Boston Dynamics AI Institute LLC.
# MIT Licence, see details in top-level file: LICENCE

"""
Classes for parameterizing a trajectory in SE3 with B-Splines. 

Copies parts of the API from scipy's B-Spline class.
"""

from typing import Any
from scipy.interpolate import BSpline
from spatialmath import SE3
import numpy as np
import matplotlib.pyplot as plt
from spatialmath.base.transforms3d import tranimate, trplot

class BSplineSE3:
    """ 
    """
    def __init__(self, knots, control_poses: list[SE3], degree) -> None:
        self.t = knots # knots 

        # a matrix where each row is a control pose as a twist
        # (so each column is a vector of control points for that dim of the twist)
        self.control_pose_matrix = np.vstack([np.array(element.twist()) for element in control_poses]) 
        
        self.k = degree # degree of spline

        self.splines = [ BSpline(knots, self.control_pose_matrix[:, i], degree) for i in range(0, 6) ]

    def __call__(self, x: float) -> Any:
        """
        x: Normalized time value to evaluate at.
        """
        twist = np.hstack([spline(x) for spline in self.splines])
        return SE3.Exp(twist)
    
    def eval(self, x):
        self.__call__(x)


def main():
    degree = 3
    control_poses = [
        SE3.Trans(
            [e, 2*np.cos(e/2 * np.pi), 2*np.sin(e/2 * np.pi)]
        )
        *SE3.Ry(e/8 * np.pi) for e in range(1,9)
    ]
    # t =  np.linspace(0, 1, len(control_poses) + degree + 1)   
    knots=np.linspace(0,1,len(control_poses)-2,endpoint=True)
    knots=np.append([0,0,0],knots)
    knots=np.append(knots,[1,1,1]) 
    trajectory = BSplineSE3(knots, control_poses, degree)
    
    out_poses = [trajectory(i) for i in np.linspace(0,1,100)]
    x = [pose.x for pose in out_poses]
    y = [pose.y for pose in out_poses]
    z = [pose.z for pose in out_poses]
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')
    trplot([np.array(control_poses)],ax=ax, length= 1.0, color="green")
    ax.plot(x,y,z)

    tranimate(
        out_poses,
        repeat=True, 
        wait=True,
        length = 1.0
    )
    

if __name__ == "__main__":
    main()