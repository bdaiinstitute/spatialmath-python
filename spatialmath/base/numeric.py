import numpy as np
from spatialmath import base

def numjac(f, x, dx=1e-8, tN=0, rN=0):
    r"""
    Numerically compute Jacobian of function

    :param f: the function, returns an m-vector
    :type f: callable
    :param x: function argument
    :type x: ndarray(n)
    :param dx: the numerical perturbation, defaults to 1e-8
    :type dx: float, optional
    :param N: function returns SE(N) matrix, defaults to 0
    :type N: int, optional
    :return: Jacobian matrix
    :rtype: ndarray(m,n)

    Computes a numerical approximation to the Jacobian for ``f(x)`` where 
    :math:`f: \mathbb{R}^n \mapsto \mathbb{R}^m`.

    Uses first-order difference :math:`J[:,i] = (f(x + dx) - f(x)) / dx`.

    If ``N`` is 2 or 3, then it is assumed that the function returns
    an SE(N) matrix which is converted into a Jacobian column comprising the
    translational Jacobian followed by the rotational Jacobian.
    """
    x = np.array(x)
    Jcol = []
    J0 = f(x)
    I = np.eye(len(x))
    f0 = np.array(f(x))
    for i in range(len(x)):
        fi = np.array(f(x + I[:,i] * dx))
        Ji = (fi - f0) / dx

        if tN > 0:
            t = Ji[:tN,tN]
            r = base.vex(Ji[:tN,:tN] @ J0[:tN,:tN].T)
            Jcol.append(np.r_[t, r])
        elif rN > 0:
            R = Ji[:rN,:rN]
            r = base.vex(R @ J0[:rN,:rN].T)
            Jcol.append(r)
        else:
            Jcol.append(Ji)
        # print(Ji)

    return np.c_[Jcol].T

