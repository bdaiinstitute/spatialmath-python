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
        fi = np.array(f(x + I[:, i] * dx))
        Ji = (fi - f0) / dx

        if SE > 0:
            t = Ji[:SE, SE]
            r = base.vex(Ji[:SE, :SE] @ J0[:SE, :SE].T)
            Jcol.append(np.r_[t, r])
        elif SO > 0:
            R = Ji[:SO, :SO]
            r = base.vex(R @ J0[:SO, :SO].T)
            Jcol.append(r)
        else:
            Jcol.append(Ji)
        # print(Ji)

    return np.c_[Jcol].T

def array2str(X, valuesep=", ", rowsep=" | ", fmt="{:.3g}", 
    brackets=("[ ", " ]"), suppress_small=True):
    """
    Convert array to single line string

    :param X: 1D or 2D array to convert
    :type X: ndarray(N,M), array_like(N)
    :param valuesep: separator between numbers, defaults to ", "
    :type valuesep: str, optional
    :param rowsep: separator between rows, defaults to " | "
    :type rowsep: str, optional
    :param format: format string, defaults to "{:.3g}"
    :type precision: str, optional
    :param brackets: strings to be added to start and end of the string, 
        defaults to ("[ ", " ]").  Set to None to suppress brackets.
    :type brackets: list, tuple of str
    :param suppress_small: small values (:math:`|x| < 10^{-12}` are converted
        to zero, defaults to True
    :type suppress_small: bool, optional
    :return: compact string representation of array
    :rtype: str

    Converts a small array to a compact single line representation.
    """
    # convert to ndarray if not already
    if isinstance(X, (list, tuple)):
        X = base.getvector(X)
    
    def format_row(x):
        s = ""
        for j, e in enumerate(x):
            if abs(e) < 1e-12:
                e = 0
            if j > 0:
                s += valuesep
            s += fmt.format(e)
        return s
    
    if X.ndim == 1:
        # 1D case
        s = format_row(X)
    else:
        # 2D case
        s = ""
        for i, row in enumerate(X):
            if i > 0:
                s += rowsep
            s += format_row(row)

    if brackets is not None and len(brackets) == 2:
        s = brackets[0] + s + brackets[1]
    return s

def bresenham(p0, p1, array=None):
    """
    Line drawing in a grid

    :param p0: initial point
    :type p0: array_like(2) of int
    :param p1: end point
    :type p1: array_like(2) of int
    :return: arrays of x and y coordinates for points along the line
    :rtype: ndarray(N), ndarray(N) of int

    Return x and y coordinate vectors for points in a grid that lie on
    a line from ``p0`` to ``p1`` inclusive.

    The end points, and all points along the line are integers.

    .. note:: The API is similar to the Bresenham algorithm but this
        implementation uses NumPy vectorised arithmetic which makes it 
        faster than the Bresenham algorithm in Python.
    """
    x0, y0 = p0
    x1, y1 = p1

    if array is not None:
        _ = array[y0, x0] + array[y1, x1]
        
    line = []

    dx = x1 - x0
    dy = y1 - y0

    if abs(dx) >= abs(dy):
        # shallow line -45° <= θ <= 45°
        # y = mx + c
        if dx == 0:
            # case p0 == p1
            x = np.r_[x0]
            y = np.r_[y0]
        else:
            m = dy / dx
            c = y0 - m * x0
            if dx > 0:
                # line to the right
                x = np.arange(x0, x1 + 1)
            elif dx < 0:
                # line to the left
                x = np.arange(x0, x1 - 1, -1)
            y = np.round(x * m + c)

    else:
        # steep line  θ < -45°,  θ > 45°
        # x = my + c
        m = dx / dy
        c = x0 - m * y0
        if dy > 0:
            # line to the right
            y = np.arange(y0, y1 + 1)
        elif dy < 0:
            # line to the left
            y = np.arange(y0, y1 - 1, -1)
        x = np.round(y * m + c)

    return x.astype(int), y.astype(int)

if __name__ == "__main__":

    print(bresenham([2,2], [2,4]))
    print(bresenham([2,2], [2,-4]))
    print(bresenham([2,2], [4,2]))
    print(bresenham([2,2], [-4,2]))
    print(bresenham([2,2], [2,2]))
    print(bresenham([2,2], [3,6])) # steep
    print(bresenham([2,2], [6,3])) # shallow
    print(bresenham([2,2], [3,6])) # steep
    print(bresenham([2,2], [6,3])) # shallow