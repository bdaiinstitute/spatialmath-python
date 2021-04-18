import numpy as np
from spatialmath import base


def numjac(f, x, dx=1e-8, SO=0, SE=0):
    r"""
    Numerically compute Jacobian of function

    :param f: the function, returns an m-vector
    :type f: callable
    :param x: function argument
    :type x: ndarray(n)
    :param dx: the numerical perturbation, defaults to 1e-8
    :type dx: float, optional
    :param SO: function returns SO(N) matrix, defaults to 0
    :type SO: int, optional
    :param SE: function returns SE(N) matrix, defaults to 0
    :type SE: int, optional

    :return: Jacobian matrix
    :rtype: ndarray(m,n)

    Computes a numerical approximation to the Jacobian for ``f(x)`` where 
    :math:`f: \mathbb{R}^n \mapsto \mathbb{R}^m`.

    Uses first-order difference :math:`J[:,i] = (f(x + dx) - f(x)) / dx`.

    If ``SO`` is 2 or 3, then it is assumed that the function returns
    an SO(N) matrix and the derivative is converted to a column vector
    
    .. math:
    
        \vex \dmat{R} \mat{R}^T
        
    If ``SE`` is 2 or 3, then it is assumed that the function returns
    an SE(N) matrix and the derivative is converted to a colun vector.
    
    """
    x = np.array(x)
    Jcol = []
    J0 = f(x)
    I = np.eye(len(x))
    f0 = np.array(f(x))
    for i in range(len(x)):
        fi = np.array(f(x + I[:,i] * dx))
        Ji = (fi - f0) / dx

        if SE > 0:
            t = Ji[:SE,SE]
            r = base.vex(Ji[:SE,:SE] @ J0[:SE,:SE].T)
            Jcol.append(np.r_[t, r])
        elif SO > 0:
            R = Ji[:SO,:SO]
            r = base.vex(R @ J0[:SO,:SO].T)
            Jcol.append(r)
        else:
            Jcol.append(Ji)
        # print(Ji)

    return np.c_[Jcol].T

