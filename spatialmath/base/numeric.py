import re
import numpy as np
from spatialmath import base
from spatialmath.base.types import *

# this is a collection of useful algorithms, not otherwise categorized


def numjac(
    f: Callable,
    x: ArrayLike,
    dx: float = 1e-8,
    SO: int = 0,
    SE: int = 0,
) -> NDArray:
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

    .. math::

        \vex{\dmat{R} \mat{R}^T}

    If ``SE`` is 2 or 3, then it is assumed that the function returns
    an SE(N) matrix and the derivative is converted to a colun vector.

    Example:

        .. runblock:: pycon

            >>> from spatialmath.base import rotx, numjac
            >>> numjac(rotx, [0])
            >>> numjac(rotx, [0], SO=3)

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


def numhess(J: Callable, x: NDArray, dx: float = 1e-8):
    r"""
    Numerically compute Hessian given Jacobian function

    :param J: the Jacobian function, returns an ndarray(m,n)
    :type J: callable
    :param x: function argument
    :type x: ndarray(n)
    :param dx: the numerical perturbation, defaults to 1e-8
    :type dx: float, optional
    :return: Hessian matrix
    :rtype: ndarray(m,n,n)

    Computes a numerical approximation to the Hessian for ``J(x)`` where
    :math:`f: \mathbb{R}^n  \mapsto \mathbb{R}^{m \times n}`.

    The result is a 3D array where

    .. math::

        H_{i,j,k} = \frac{\partial J_{j,k}}{\partial x_i}

    Uses first-order difference :math:`H[:,:,i] = (J(x + dx) - J(x)) / dx`.
    """

    I = np.eye(len(x))
    Hcol = []
    J0 = J(x)
    for i in range(len(x)):
        Ji = J(x + I[:, i] * dx)
        Hi = (Ji - J0) / dx

        Hcol.append(Hi)

    return np.stack(Hcol, axis=0)


def array2str(
    X: NDArray,
    valuesep: str = ", ",
    rowsep: str = " | ",
    fmt: str = "{:.3g}",
    brackets: Tuple[str, str] = ("[ ", " ]"),
    suppress_small: bool = True,
) -> str:
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

    Example:

    .. runblock:: pycon

        >>> from spatialmath.base import array2str
        >>> import numpy as np
        >>> array2str(np.random.rand(2,2))
        >>> array2str(np.random.rand(2,2), rowsep="; ")  # MATLAB-like
        >>> array2str(np.random.rand(3,))
        >>> array2str(np.random.rand(3,1))


    :seealso: :func:`array2str`
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


def str2array(s: str) -> NDArray:
    """
    Convert compact single line string to array

    :param s: string to convert
    :type s: str
    :return: array
    :rtype: ndarray

    Convert a string containing a "MATLAB-like" matrix definition to a NumPy
    array.  A scalar has no delimiting square brackets and becomes a 1x1 array.
    A 2D array is delimited by square brackets, elements are separated by a comma,
    and rows are separated by a semicolon.  Extra white spaces are ignored.

    Example:

    .. runblock:: pycon

        >>> from spatialmath.base import str2array
        >>> str2array("5")
        >>> str2array("[1 2 3]")
        >>> str2array("[1 2; 3 4]")
        >>> str2array(" [  1  , 2 ; 3 4  ] ")
        >>> str2array("[1; 2; 3]")

    :seealso: :func:`array2str`
    """

    s = s.lstrip(" [")
    s = s.rstrip(" ]")
    values = []
    for row in s.split(";"):
        values.append([float(x) for x in re.split("[, ]+", row.strip())])
    return np.array(values)


def bresenham(p0: ArrayLike2, p1: ArrayLike2) -> Tuple[NDArray, NDArray]:
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

    * The end points, and all points along the line are integers.
    * Points are always adjacent, but the slope from point to point is not constant.


    Example:

    .. runblock:: pycon

        >>> from spatialmath.base import bresenham
        >>> bresenham((2, 4), (10, 10))

    .. plot::

        from spatialmath.base import bresenham
        import matplotlib.pyplot as plt
        p = bresenham((2, 4), (10, 10))
        plt.plot((2, 10), (4, 10))
        plt.plot(p[0], p[1], 'ok')
        plt.plot(p[0], p[1], 'k', drawstyle='steps-post')
        ax = plt.gca()
        ax.grid()


    .. note:: The API is similar to the Bresenham algorithm but this
        implementation uses NumPy vectorised arithmetic which makes it
        faster than the Bresenham algorithm in Python.
    """
    x0, y0 = p0
    x1, y1 = p1

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


def mpq_point(data: Points2, p: int, q: int) -> float:
    r"""
    Moments of polygon

    :param data: polygon vertices, points as columns
    :type data: ndarray(2,N)
    :param p: moment order x
    :type p: int
    :param q: moment order y
    :type q: int

    Returns the pq'th moment of the polygon

    .. math::

        M(p, q) = \sum_{i=0}^{n-1} x_i^p y_i^q

    Example:

    .. runblock:: pycon

        >>> from spatialmath.base import mpq_point
        >>> import numpy as np
        >>> p = np.array([[1, 3, 2], [2, 2, 4]])
        >>> mpq_point(p, 0, 0)  # area
        >>> mpq_point(p, 3, 0)

    .. note:: is negative for clockwise perimeter.
    """
    x = data[0, :]
    y = data[1, :]

    return np.sum(x**p * y**q)


def gauss1d(mu: float, var: float, x: ArrayLike):
    """
    Gaussian function in 1D

    :param mu: mean
    :type mu: float
    :param var: variance
    :type var: float
    :param x: x-coordinate values
    :type x: array_like(n)
    :return: Gaussian :math:`G(x)`
    :rtype: ndarray(n)

    Example::

        >>> g = gauss1d(5, 2, np.linspace(0, 10, 100))

    .. plot::

        from spatialmath.base import gauss1d
        import matplotlib.pyplot as plt
        import numpy as np
        x = np.linspace(0, 10, 100)
        g = gauss1d(5, 2, x)
        plt.plot(x, g)
        plt.grid()

    :seealso: :func:`gauss2d`
    """
    sigma = np.sqrt(var)
    x = base.getvector(x)

    return (
        1.0
        / np.sqrt(sigma**2 * 2 * np.pi)
        * np.exp(-((x - mu) ** 2) / 2 / sigma**2)
    )


def gauss2d(mu: ArrayLike2, P: NDArray, X: NDArray, Y: NDArray) -> NDArray:
    """
    Gaussian function in 2D

    :param mu: mean
    :type mu: array_like(2)
    :param P: covariance matrix
    :type P: ndarray(2,2)
    :param X: array of x-coordinates
    :type X:  ndarray(n,m)
    :param Y: array of y-coordinates
    :type Y: ndarray(n,m)
    :return: Gaussian :math:`g(x,y)`
    :rtype: ndarray(n,m)

    Computed :math:`g_{i,j} = G(x_{i,j}, y_{i,j})`

    Example (RVC3 Fig G.2)::

        >>> a = np.linspace(-5, 5, 100)
        >>> X, Y = np.meshgrid(a, a)
        >>> P = np.diag([1, 2])**2;
        >>> g = gauss2d(X, Y, [0, 0], P)

    .. plot::

        from spatialmath.base import gauss2d, plotvol3
        import matplotlib.pyplot as plt
        import numpy as np
        a = np.linspace(-5, 5, 100)
        x, y = np.meshgrid(a, a)
        P = np.diag([1, 2])**2;
        g = gauss2d([0, 0], P, x, y)
        ax = plotvol3()
        ax.plot_surface(x, y, g)

    :seealso: :func:`gauss1d`
    """

    x = X.ravel() - mu[0]
    y = Y.ravel() - mu[1]

    Pi = np.linalg.inv(P)
    g = (
        1
        / (2 * np.pi * np.sqrt(np.linalg.det(P)))
        * np.exp(-0.5 * (x**2 * Pi[0, 0] + y**2 * Pi[1, 1] + 2 * x * y * Pi[0, 1]))
    )
    return g.reshape(X.shape)


if __name__ == "__main__":
    r = np.linspace(-4, 4, 6)
    x, y = np.meshgrid(r, r)
    print(gauss2d([0, 0], np.diag([1, 2]), x, y))
    # print(bresenham([2,2], [2,4]))
    # print(bresenham([2,2], [2,-4]))
    # print(bresenham([2,2], [4,2]))
    # print(bresenham([2,2], [-4,2]))
    # print(bresenham([2,2], [2,2]))
    # print(bresenham([2,2], [3,6])) # steep
    # print(bresenham([2,2], [6,3])) # shallow
    # print(bresenham([2,2], [3,6])) # steep
    # print(bresenham([2,2], [6,3])) # shallow
