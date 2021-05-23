import math
from itertools import product
import warnings
import numpy as np
import scipy as sp
from scipy.stats.distributions import chi2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d.art3d import (
    Poly3DCollection,
    Line3DCollection,
    pathpatch_2d_to_3d,
)
from spatialmath import base

"""
Set of functions to draw 2D and 3D graphical primitives using matplotlib.

The 2D functions all allow color and line style to be specified by a fmt string
like, 'r' or 'b--'.

The 3D functions require explicity arguments to set properties, like color='b'

All return a list of the graphic objects they create.

"""
# TODO
# return a redrawer object, that can be used for animation

# =========================== 2D shapes =================================== #


def plot_text(pos, text=None, ax=None, color=None, **kwargs):
    """
    Plot text using matplotlib

    :param pos: position of text
    :type pos: array_like(2)
    :param text: text
    :type text: str
    :param ax: axes to draw in, defaults to ``gca()``
    :type ax: Axis, optional
    :param color: text color, defaults to None
    :type color: str or array_like(3), optional
    :param kwargs: additional arguments passed to ``pyplot.text()``
    :return: the matplotlib object
    :rtype: list of Text instance

    Example:

    .. runblock:: pycon

        >>> from spatialmath.base import plotvol2, plot_text
        >>> plotvol2(5)
        >>> plot_text((1,3), 'foo')
        >>> plot_text((2,2), 'bar', 'b')
        >>> plot_text((2,2), 'baz', fontsize=14, horizontalalignment='centre')
    """

    defaults = {"horizontalalignment": "left", "verticalalignment": "center"}
    for k, v in defaults.items():
        if k not in kwargs:
            kwargs[k] = v
    if ax is None:
        ax = plt.gca()

    handle = plt.text(pos[0], pos[1], text, color=color, **kwargs)
    return [handle]


def plot_point(pos, marker="bs", text=None, ax=None, textargs=None, **kwargs):
    """
    Plot a point using matplotlib

    :param pos: position of marker
    :type pos: array_like(2), ndarray(2,n), list of 2-tuples
    :param marker: matplotlub marker style, defaults to 'bs'
    :type marker: str or list of str, optional
    :param text: text label, defaults to None
    :type text: str, optional
    :param ax: axes to plot in, defaults to ``gca()``
    :type ax: Axis, optional
    :return: the matplotlib object
    :rtype: list of Text and Line2D instances

    Plot one or more points, with optional text label.

    - The color of the marker can be different to the color of the text, the
      marker color is specified by a single letter in the marker string.

    - A point can have multiple markers, given as a list, which will be
      overlaid, for instance ``["rx", "ro"]`` will give a ⨂ symbol.

    - The optional text label is placed to the right of the marker, and
      vertically aligned.

    - Multiple points can be marked if ``pos`` is a 2xn array or a list of
      coordinate pairs.  If a label is provided every point will have the same
      label.

    Examples:

    - ``plot_point((1,2))`` plot default marker at coordinate (1,2)
    - ``plot_point((1,2), 'r*')`` plot red star at coordinate (1,2)
    - ``plot_point((1,2), 'r*', 'foo')`` plot red star at coordinate (1,2) and
      label it as 'foo'
    - ``plot_point(p, 'r*')`` plot red star at points defined by columns of
      ``p``.
    - ``plot_point(p, 'r*', 'foo')`` plot red star at points defined by columns
      of ``p`` and label them all as 'foo'
    - ``plot_point(p, 'r*', '{0}')`` plot red star at points defined by columns
      of ``p`` and label them sequentially from 0
    - ``plot_point(p, 'r*', ('{1:.1f}', z))`` plot red star at points defined by
      columns of ``p`` and label them all with successive elements of ``z``.
    """
    if isinstance(pos, np.ndarray):
        if pos.ndim == 1:
            x = pos[0]
            y = pos[1]
        elif pos.ndim == 2 and pos.shape[0] == 2:
            x = pos[0, :]
            y = pos[1, :]
    elif isinstance(pos, (tuple, list)):
        # [x, y]
        # [(x,y), (x,y), ...]
        # [xlist, ylist]
        # [xarray, yarray]
        if base.islistof(pos, (tuple, list)):
            x = [z[0] for z in pos]
            y = [z[1] for z in pos]
        elif base.islistof(pos, np.ndarray):
            x = pos[0]
            y = pos[1]
        else:
            x = pos[0]
            y = pos[1]

    textopts = {
        "fontsize": 12,
        "horizontalalignment": "left",
        "verticalalignment": "center",
    }
    if textargs is not None:
        textopts = {**textopts, **textargs}

    if ax is None:
        ax = plt.gca()

    handles = []
    if isinstance(marker, (list, tuple)):
        for m in marker:
            handles.append(plt.plot(x, y, m, **kwargs))
    else:
        handles.append(plt.plot(x, y, marker, **kwargs))
    if text is not None:
        if isinstance(text, str):
            # simple string, but might have format chars
            for i, xy in enumerate(zip(x, y)):
                handles.append(plt.text(xy[0], xy[1], " " + text.format(i), **textopts))
        elif isinstance(text, (tuple, list)):
            for i, xy in enumerate(zip(x, y)):
                handles.append(
                    plt.text(
                        xy[0],
                        xy[1],
                        " " + text[0].format(i, *[d[i] for d in text[1:]]),
                        **textopts
                    )
                )
    return handles


def plot_homline(lines, *args, ax=None, **kwargs):
    """
    Plot a homogeneous line using matplotlib

    :param lines: homgeneous lines
    :type lines: array_like(3), ndarray(3,N)
    :param ax: axes to plot in, defaults to ``gca()``
    :type ax: Axis, optional
    :param kwargs: arguments passed to ``plot``
    :return: matplotlib object
    :rtype: list of Line2D instances

    Draws the 2D line given in homogeneous form :math:`\ell[0] x + \ell[1] y + \ell[2] = 0` in the current
    2D axes.

    .. warning: A set of 2D axes must exist in order that the axis limits can
        be obtained. The line is drawn from edge to edge.

    If ``lines`` is a 3xN array then ``N`` lines are drawn, one per column.

    Example:

    .. runblock:: pycon

        >>> from spatialmath.base import plotvol2, plot_homline
        >>> plotvol2(5)
        >>> plot_homline((1, -2, 3))
        >>> plot_homline((1, -2, 3), 'k--') # dashed black line
    """
    ax = axes_logic(ax, 2)
    # get plot limits from current graph
    xlim = np.r_[ax.get_xlim()]
    ylim = np.r_[ax.get_ylim()]

    lines = base.getmatrix(lines, (None, 3))

    handles = []
    for line in lines:
        if abs(line[1]) > abs(line[0]):
            y = (-line[2] - line[0] * xlim) / line[1]
            ax.plot(xlim, y, *args, **kwargs)
        else:
            x = (-line[2] - line[1] * ylim) / line[0]
            handles.append(ax.plot(x, ylim, *args, **kwargs))

    return handles


def plot_box(
    *fmt,
    bl=None,
    tl=None,
    br=None,
    tr=None,
    wh=None,
    centre=None,
    l=None,
    r=None,
    t=None,
    b=None,
    w=None,
    h=None,
    ax=None,
    bbox=None,
    filled=False,
    **kwargs
):
    """
    Plot a 2D box using matplotlib

    :param bl: bottom-left corner, defaults to None
    :type bl: array_like(2), optional
    :param tl: top-left corner, defaults to None
    :type tl: [array_like(2), optional
    :param br: bottom-right corner, defaults to None
    :type br: array_like(2), optional
    :param tr: top -ight corner, defaults to None
    :type tr: array_like(2), optional
    :param wh: width and height, defaults to None
    :type wh: array_like(2), optional
    :param centre: centre of box, defaults to None
    :type centre: array_like(2), optional
    :param l: left side of box, minimum x, defaults to None
    :type l: float, optional
    :param r: right side of box, minimum x, defaults to None
    :type r: float, optional
    :param b: bottom side of box, minimum y, defaults to None
    :type b: float, optional
    :param t: top side of box, maximum y, defaults to None
    :type t: float, optional
    :param w: width of box, defaults to None
    :type w: float, optional
    :param h: height of box, defaults to None
    :type h: float, optional
    :param ax: the axes to draw on, defaults to ``gca()``
    :type ax: Axis, optional
    :param bbox: bounding box matrix, defaults to None
    :type bbox: ndarray(2,2), optional
    :param color: box outline color
    :type color: array_like(3) or str
    :param fillcolor: box fill color
    :type fillcolor: array_like(3) or str
    :param alpha: transparency, defaults to 1
    :type alpha: float, optional
    :param thickness: line thickness, defaults to None
    :type thickness: float, optional
    :return: the matplotlib object
    :rtype: list of Line2D or Patch.Rectangle instance

    The box can be specified in many ways:

    - bounding box which is a 2x2 matrix [xmin, xmax; ymin, ymax]
    - centre and width+height
    - bottom-left and top-right corners
    - bottom-left corner and width+height
    - top-right corner and width+height
    - top-left corner and width+height

    Example:

    .. runblock:: pycon

        >>> from spatialmath.base import plotvol2, plot_box
        >>> plotvol2(5)
        >>> plot_box('r', centre=(2,3), wh=(1,1))
        >>> plot_box(tl=(1,1), br=(0,2), filled=True, color='b')
    """

    if bbox is not None:
        l, r, b, t = bbox
    else:
        if tl is not None:
            l, t = tl
        if tr is not None:
            r, t = tr
        if bl is not None:
            l, b = bl
        if br is not None:
            r, b = br
        if wh is not None:
            w, h = wh
        if centre is not None:
            cx, cy = centre
        if l is None:
            try:
                l = r - w
            except:
                pass
        if l is None:
            try:
                l = cx - w / 2
            except:
                pass
        if b is None:
            try:
                b = t - h
            except:
                pass
        if b is None:
            try:
                b = cy + h / 2
            except:
                pass

    ax = axes_logic(ax, 2)

    if filled:
        if w is None:
            try:
                w = r - l
            except:
                pass
        if h is None:
            try:
                h = t - b
            except:
                pass
        r = plt.Rectangle((l, b), w, h, clip_on=True, **kwargs)
        ax.add_patch(r)
    else:
        if r is None:
            try:
                r = l + w
            except:
                pass
        if r is None:
            try:
                l = cx + w / 2
            except:
                pass
        if t is None:
            try:
                t = b + h
            except:
                pass
        if t is None:
            try:
                t = cy + h / 2
            except:
                pass
        r = plt.plot([l, l, r, r, l], [b, t, t, b, b], *fmt, **kwargs)

    return [r]


def circle(centre=(0, 0), radius=1, resolution=50):
    """
    Points on a circle

    :param centre: centre of circle, defaults to (0, 0)
    :type centre: array_like(2), optional
    :param radius: radius of circle, defaults to 1
    :type radius: float, optional
    :param resolution: number of points on circumferece, defaults to 50
    :type resolution: int, optional
    :return: points on circumference
    :rtype: ndarray(2,N)

    Returns a set of ``resolution`` that lie on the circumference of a circle
    of given ``center`` and ``radius``.
    """
    u = np.linspace(0.0, 2.0 * np.pi, resolution)
    x = radius * np.cos(u) + centre[0]
    y = radius * np.sin(u) + centre[1]

    return np.array((x, y))


def plot_circle(
    radius, *fmt, centre=(0, 0), resolution=50, ax=None, filled=False, **kwargs
):
    """
    Plot a circle using matplotlib

    :param centre: centre of circle, defaults to (0,0)
    :type centre: array_like(2), optional
    :param args:
    :param radius: radius of circle
    :type radius: float
    :param resolution: number of points on circumferece, defaults to 50
    :type resolution: int, optional
    :return: the matplotlib object
    :rtype: list of Line2D or Patch.Polygon

    Plot or more circles. If ``centre`` is a 3xN array, then each column is
    taken as the centre of a circle.  All circles have the same radius, color
    etc.

    Example:

    .. runblock:: pycon

        >>> from spatialmath.base import plotvol2, plot_circle
        >>> plotvol2(5)
        >>> plot_circle(1, 'r')  # red circle
        >>> plot_circle(2, 'b--')  # blue dashed circle
        >>> plot_circle(0.5, filled=True, facecolor='y')  # yellow filled circle
    """
    centres = base.getmatrix(centre, (2, None))

    ax = axes_logic(ax, 2)
    handles = []
    for centre in centres.T:
        xy = circle(centre, radius, resolution)
        if filled:
            patch = plt.Polygon(xy.T, **kwargs)
            handles.append(ax.add_patch(patch))
        else:
            handles.append(ax.plot(xy[0, :], xy[1, :], *fmt, **kwargs))
    return handles


def ellipse(E, centre=(0, 0), scale=1, confidence=None, resolution=40, inverted=False):
    """
    Points on ellipse

    :param E: ellipse
    :type E: ndarray(2,2)
    :param centre: ellipse centre, defaults to (0,0,0)
    :type centre: tuple, optional
    :param scale: scale factor for the ellipse radii
    :type scale: float
    :param confidence: if E is an inverse covariance matrix plot an ellipse
        for this confidence interval in the range [0,1], defaults to None
    :type confidence: float, optional
    :param resolution: number of points on circumferance, defaults to 40
    :type resolution: int, optional
    :param inverted: if :math:`\mat{E}^{-1}` is provided, defaults to False
    :type inverted: bool, optional
    :raises ValueError: [description]
    :return: points on circumference
    :rtype: ndarray(2,N)

    The ellipse is defined by :math:`x^T \mat{E} x = s^2` where :math:`x \in
    \mathbb{R}^2` and :math:`s` is the scale factor.

    .. note:: For some common cases we require :math:`\mat{E}^{-1}`, for example
        - for robot manipulability
          :math:`\nu (\mat{J} \mat{J}^T)^{-1} \nu` i
        - a covariance matrix
          :math:`(x - \mu)^T \mat{P}^{-1} (x - \mu)`
        so to avoid inverting ``E`` twice to compute the ellipse, we flag that
        the inverse is provided using ``inverted``.
    """
    if E.shape != (2, 2):
        raise ValueError("ellipse is defined by a 2x2 matrix")

    if confidence:
        # process the probability
        s = math.sqrt(chi2.ppf(confidence, df=2)) * scale
    else:
        s = scale

    xy = circle(resolution=resolution)  # unit circle

    if not inverted:
        E = np.linalg.inv(E)

    e = s * sp.linalg.sqrtm(E) @ xy + np.array(centre, ndmin=2).T
    return e


def plot_ellipse(
    E,
    *fmt,
    centre=(0, 0),
    scale=1,
    confidence=None,
    resolution=40,
    inverted=False,
    ax=None,
    filled=None,
    **kwargs
):
    """
    Plot an ellipse using matplotlib

    :param E: matrix describing ellipse
    :type E: ndarray(2,2)
    :param centre: centre of ellipse, defaults to (0, 0)
    :type centre: array_like(2), optional
    :param scale: scale factor for the ellipse radii
    :type scale: float
    :param resolution: number of points on circumferece, defaults to 40
    :type resolution: int, optional
    :return: the matplotlib object
    :rtype: Line2D or Patch.Polygon

    The ellipse is defined by :math:`x^T \mat{E} x = s^2` where :math:`x \in
    \mathbb{R}^2` and :math:`s` is the scale factor.

    .. note:: For some common cases we require :math:`\mat{E}^{-1}`, for example
        - for robot manipulability
          :math:`\nu (\mat{J} \mat{J}^T)^{-1} \nu` i
        - a covariance matrix
          :math:`(x - \mu)^T \mat{P}^{-1} (x - \mu)`
        so to avoid inverting ``E`` twice to compute the ellipse, we flag that
        the inverse is provided using ``inverted``.

    Returns a set of ``resolution``  that lie on the circumference of a circle
    of given ``center`` and ``radius``.

    Example:

    .. runblock:: pycon

        >>> from spatialmath.base import plotvol2, plot_circle
        >>> plotvol2(5)
        >>> plot_ellipse(np.diag((1,2)), 'r')  # red ellipse
        >>> plot_ellipse(np.diag((1,2)), 'b--')  # blue dashed ellipse
        >>> plot_ellipse(np.diag((1,2)), filled=True, facecolor='y')  # yellow filled ellipse

    """
    # allow for centre[2] to plot ellipse in a plane in a 3D plot

    xy = ellipse(E, centre, scale, confidence, resolution, inverted)
    ax = axes_logic(ax, 2)
    if filled:
        patch = plt.Polygon(xy.T, **kwargs)
        ax.add_patch(patch)
    else:
        plt.plot(xy[0, :], xy[1, :], *fmt, **kwargs)


# =========================== 3D shapes =================================== #


def sphere(radius=1, centre=(0, 0, 0), resolution=50):
    """
    Points on a sphere

    :param centre: centre of sphere, defaults to (0, 0, 0)
    :type centre: array_like(3), optional
    :param radius: radius of sphere, defaults to 1
    :type radius: float, optional
    :param resolution: number of points ``N`` on circumferece, defaults to 50
    :type resolution: int, optional
    :return: X, Y and Z braid matrices
    :rtype: 3 x ndarray(N, N)

    :seealso: :func:`plot_sphere`, :func:`~matplotlib.pyplot.plot_surface`, :func:`~matplotlib.pyplot.plot_wireframe`
    """
    u = np.linspace(0.0, 2.0 * np.pi, resolution)
    v = np.linspace(0.0, np.pi, resolution)

    x = radius * np.outer(np.cos(u), np.sin(v)) + centre[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + centre[1]
    z = radius * np.outer(np.ones_like(u), np.cos(v)) + centre[2]

    return (x, y, z)


def plot_sphere(radius, centre=(0, 0, 0), pose=None, resolution=50, ax=None, **kwargs):
    """
    Plot a sphere using matplotlib

    :param centre: centre of sphere, defaults to (0, 0, 0)
    :type centre: array_like(3), ndarray(3,N), optional
    :param radius: radius of sphere, defaults to 1
    :type radius: float, optional
    :param resolution: number of points on circumferece, defaults to 50
    :type resolution: int, optional

    :param pose: pose of sphere, defaults to None
    :type pose: SE3, optional
    :param ax: axes to draw into, defaults to None
    :type ax: Axes3D, optional
    :param filled: draw filled polygon, else wireframe, defaults to False
    :type filled: bool, optional
    :param kwargs: arguments passed to ``plot_wireframe`` or ``plot_surface``

    :return: matplotlib collection
    :rtype: list of Line3DCollection or Poly3DCollection

    Plot one or more spheres. If ``centre`` is a 3xN array, then each column is
    taken as the centre of a sphere.  All spheres have the same radius, color
    etc.

    Example:

    .. runblock:: pycon

        >>> from spatialmath.base import plot_sphere
        >>> plot_sphere(1, 'r')   # red sphere wireframe
        >>> plot_sphere(1, centre=(1,1,1), filled=True, facecolor='b')


    :seealso: :func:`~matplotlib.pyplot.plot_surface`, :func:`~matplotlib.pyplot.plot_wireframe`
    """
    ax = axes_logic(ax, 3)

    centre = base.getmatrix(centre, (3, None))

    handles = []
    for c in centre.T:
        X, Y, Z = sphere(centre=c, radius=radius, resolution=resolution)

        if pose is not None:
            xc = X.reshape((-1,))
            yc = Y.reshape((-1,))
            zc = Z.reshape((-1,))
            xyz = np.array((xc, yc, zc))
            xyz = pose * xyz
            X = xyz[0, :].reshape(x.shape)
            Y = xyz[1, :].reshape(y.shape)
            Z = xyz[2, :].reshape(z.shape)

        handles.append(_render3D(ax, X, Y, Z, **kwargs))

    return handles


def ellipsoid(
    E, centre=(0, 0, 0), scale=1, confidence=None, resolution=40, inverted=False
):
    """
    Points on an ellipsoid

    :param centre: centre of ellipsoid, defaults to (0, 0, 0)
    :type centre: array_like(3), optional
    :param scale: scale factor for the ellipse radii
    :type scale: float
    :param confidence: confidence interval, range 0 to 1
    :type confidence: float
    :param resolution: number of points ``N`` on circumferece, defaults to 40
    :type resolution: int, optional
    :param inverted: :math:`E^{-1}` rather than :math:`E` provided, defaults to False
    :type inverted: bool, optional
    :return: X, Y and Z braid matrices
    :rtype: 3 x ndarray(N, N)

    The ellipse is defined by :math:`x^T \mat{E} x = s^2` where :math:`x \in
    \mathbb{R}^3` and :math:`s` is the scale factor.

    .. note:: For some common cases we require :math:`\mat{E}^{-1}`, for example
        - for robot manipulability
          :math:`\nu (\mat{J} \mat{J}^T)^{-1} \nu` i
        - a covariance matrix
          :math:`(x - \mu)^T \mat{P}^{-1} (x - \mu)`
        so to avoid inverting ``E`` twice to compute the ellipse, we flag that
        the inverse is provided using ``inverted``.

    :seealso: :func:`plot_ellipsoid`, :func:`~matplotlib.pyplot.plot_surface`, :func:`~matplotlib.pyplot.plot_wireframe`
    """
    if E.shape != (3, 3):
        raise ValueError("ellipsoid is defined by a 3x3 matrix")

    if confidence:
        # process the probability
        from scipy.stats.distributions import chi2

        s = math.sqrt(chi2.ppf(confidence, df=2)) * scale
    else:
        s = scale

    if not inverted:
        E = np.linalg.inv(E)

    x, y, z = sphere()  # unit sphere
    e = (
        s * sp.linalg.sqrtm(E) @ np.array([x.flatten(), y.flatten(), z.flatten()])
        + np.c_[centre].T
    )
    return e[0, :].reshape(x.shape), e[1, :].reshape(x.shape), e[2, :].reshape(x.shape)


def plot_ellipsoid(
    E,
    centre=(0, 0, 0),
    scale=1,
    confidence=None,
    resolution=40,
    inverted=False,
    ax=None,
    **kwargs
):
    """
    Draw an ellipsoid using matplotlib

    :param E: ellipsoid
    :type E: ndarray(3,3)
    :param centre: [description], defaults to (0,0,0)
    :type centre: tuple, optional
    :param scale:
    :type scale:
    :param confidence: confidence interval, range 0 to 1
    :type confidence: float
    :param resolution: number of points on circumferece, defaults to 40
    :type resolution: int, optional
    :param inverted: :math:`E^{-1}` rather than :math:`E` provided, defaults to False
    :type inverted: bool, optional
    :param ax: [description], defaults to None
    :type ax: [type], optional
    :param wireframe: [description], defaults to False
    :type wireframe: bool, optional
    :param stride: [description], defaults to 1
    :type stride: int, optional

    ``plot_ellipse(E)`` draws the ellipsoid defined by :math:`x^T \mat{E} x = 0`
    on the current plot.

    Example:

          H = plot_ellipse(diag([1 2]), [3 4]', 'r'); % draw red ellipse
          plot_ellipse(diag([1 2]), [5 6]', 'alter', H); % move the ellipse
          plot_ellipse(diag([1 2]), [5 6]', 'alter', H, 'LineColor', 'k'); % change color

          plot_ellipse(COVAR, 'confidence', 0.95); % draw 95% confidence ellipse

    .. note::

        - If a confidence interval is given then ``E`` is interpretted as a covariance
          matrix and the ellipse size is computed using an inverse chi-squared function.

    :seealso: :func:`~matplotlib.pyplot.plot_surface`, :func:`~matplotlib.pyplot.plot_wireframe`
    """
    X, Y, Z = ellipsoid(E, centre, scale, confidence, resolution, inverted)
    ax = axes_logic(ax, 3)
    handle = _render3D(ax, X, Y, Z, **kwargs)
    return [handle]


def plot_cylinder(
    radius,
    height,
    resolution=50,
    centre=(0, 0, 0),
    ends=False,
    ax=None,
    filled=False,
    **kwargs
):
    """
    Plot a cylinder using matplotlib

    :param radius: radius of sphere, defaults to 1
    :type radius: float, optional
    :param height: height of cylinder in the z-direction
    :type height: float or array_like(2)
    :param resolution: number of points on circumferece, defaults to 50
    :type resolution: int, optional

    :param pose: pose of sphere, defaults to None
    :type pose: SE3, optional
    :param ax: axes to draw into, defaults to None
    :type ax: Axes3D, optional
    :param filled: draw filled polygon, else wireframe, defaults to False
    :type filled: bool, optional
    :param kwargs: arguments passed to ``plot_wireframe`` or ``plot_surface``

    :return: matplotlib objects
    :rtype: list of matplotlib object types

    The axis of the cylinder is parallel to the z-axis and extends from z=0
    to z=height, or z=height[0] to z=height[1].

    The cylinder can be positioned by setting ``centre``, or positioned
    and orientated by setting ``pose``.

    :seealso: :func:`~matplotlib.pyplot.plot_surface`, :func:`~matplotlib.pyplot.plot_wireframe`
    """
    if base.isscalar(height):
        height = [0, height]

    ax = axes_logic(ax, 3)
    x = np.linspace(centre[0] - radius, centre[0] + radius, resolution)
    z = height
    X, Z = np.meshgrid(x, z)

    Y = np.sqrt(radius ** 2 - (X - centre[0]) ** 2) + centre[1]  # Pythagorean theorem

    handles = []
    handles.append(_render3D(ax, X, Y, Z, filled=filled, **kwargs))
    handles.append(_render3D(ax, X, (2 * centre[1] - Y), Z, filled=filled, **kwargs))

    if ends and kwargs.get("filled", default=False):
        floor = Circle(centre[:2], radius, **kwargs)
        handles.append(ax.add_patch(floor))
        pathpatch_2d_to_3d(floor, z=height[0], zdir="z")

        ceiling = Circle(centre[:2], radius, **kwargs)
        handles.append(ax.add_patch(ceiling))
        pathpatch_2d_to_3d(ceiling, z=height[1], zdir="z")

    return handles


def plot_cuboid(
    sides=[1, 1, 1], centre=(0, 0, 0), pose=None, ax=None, filled=False, **kwargs
):
    """
    Plot a cuboid (3D box) using matplotlib

    :param sides: side lengths, defaults to 1
    :type sides: array_like(3), optional
    :param centre: centre of box, defaults to (0, 0, 0)
    :type centre: array_like(3), optional

    :param pose: pose of sphere, defaults to None
    :type pose: SE3, optional
    :param ax: axes to draw into, defaults to None
    :type ax: Axes3D, optional
    :param filled: draw filled polygon, else wireframe, defaults to False
    :type filled: bool, optional
    :param kwargs: arguments passed to ``plot_wireframe`` or ``plot_surface``

    :return: matplotlib collection
    :rtype: Line3DCollection or Poly3DCollection

    :seealso: :func:`~matplotlib.pyplot.plot_surface`, :func:`~matplotlib.pyplot.plot_wireframe`
    """

    vertices = (
        np.array(
            list(
                product(
                    [-sides[0], sides[0]], [-sides[1], sides[1]], [-sides[2], sides[2]]
                )
            )
        )
        / 2
        + centre
    )
    vertices = vertices.T

    if pose is not None:
        vertices = pose * vertices

    ax = axes_logic(ax, 3)
    # plot sides
    if filled:
        # show faces

        faces = [
            [0, 1, 3, 2],
            [4, 5, 7, 6],  # YZ planes
            [0, 1, 5, 4],
            [2, 3, 7, 6],  # XZ planes
            [0, 2, 6, 4],
            [1, 3, 7, 5],  # XY planes
        ]
        F = [[vertices[:, i] for i in face] for face in faces]
        collection = Poly3DCollection(F, **kwargs)
        ax.add_collection3d(collection)
        return collection
    else:
        edges = [[0, 1, 3, 2, 0], [4, 5, 7, 6, 4], [0, 4], [1, 5], [3, 7], [2, 6]]
        lines = []
        for edge in edges:
            E = vertices[:, edge]
            # ax.plot(E[0], E[1], E[2], **kwargs)
            lines.append(E.T)
        collection = Line3DCollection(lines, **kwargs)
        ax.add_collection3d(collection)
        return collection


def _render3D(ax, X, Y, Z, pose=None, filled=False, color=None, **kwargs):

    # TODO:
    # handle pose in here
    # do the guts of plot_surface/wireframe but skip the auto scaling
    # have all 3d functions use this
    # rename all functions with 3d suffix sphere3d, box3d, ell

    if pose is not None:
        # long version:
        # xc = X.reshape((-1,))
        # yc = Y.reshape((-1,))
        # zc = Z.reshape((-1,))
        # xyz = np.array((xc, yc, zc))
        # xyz = pose * xyz
        # X = xyz[0, :].reshape(X.shape)
        # Y = xyz[1, :].reshape(Y.shape)
        # Z = xyz[2, :].reshape(Z.shape)

        # short version:
        xyz = pose * np.dstack((X, Y, Z)).reshape((-1, 3)).T
        X, Y, Z = np.squeeze(np.dsplit(xyz.T.reshape(X.shape + (3,)), 3))

    if filled:
        ax.plot_surface(X, Y, Z, color=color, **kwargs)
    else:
        kwargs["colors"] = color
        ax.plot_wireframe(X, Y, Z, **kwargs)


def _axes_dimensions(ax):
    """
    Dimensions of axes

    :param ax: axes
    :type ax: Axes3DSubplot or AxesSubplot
    :return: dimensionality of axes, either 2 or 3
    :rtype: int
    """
    classname = ax.__class__.__name__

    if classname == "Axes3DSubplot":
        return 3
    elif classname == "AxesSubplot":
        return 2


def axes_logic(ax, dimensions, projection="ortho"):
    """
    Axis creation logic

    :param ax: axes to draw in
    :type ax: Axes3DSubplot, AxesSubplot or None
    :param dimensions: required dimensionality, 2 or 3
    :type dimensions: int
    :param projection: 3D projection type, defaults to 'ortho'
    :type projection: str, optional
    :return: axes to draw in
    :rtype: Axes3DSubplot or AxesSubplot

    Given a request for axes with either 2 or 3 dimensions it checks for a
    match with the passed axes ``ax`` or the current axes.

    If the dimensions do not match, or no figure/axes currently exist,
    then ``plt.axes()`` is called to create one.

    Used by all plot_xxx() functions in this module.
    """
    # print(f"new axis logic ({dimensions}D): ", end='')
    if ax is None:
        # no axes passed in, find out what's happening
        # need to be careful to not use gcf() or gca() since they
        # auto create fig/axes if none exist
        nfigs = len(plt.get_fignums())
        if nfigs > 0:
            # there are figures
            fig = plt.gcf()  # get current figure
            naxes = len(fig.axes)
            # print(f"existing fig with {naxes} axes")
            if naxes > 0:
                ax = plt.gca()  # get current axes
                if _axes_dimensions(ax) == dimensions:
                    return ax
        # otherwise it doesnt exist or dimension mismatch, create new axes

    else:
        # axis was given

        if _axes_dimensions(ax) == dimensions:
            print("use existing axes")
            return ax
        # mismatch in dimensions, create new axes
    # print('create new axes')
    plt.figure()
    # no axis specified
    if dimensions == 2:
        ax = plt.axes()
    else:
        ax = plt.axes(projection="3d", proj_type=projection)
    return ax


def plotvol2(dim, ax=None, equal=True, grid=False):
    """
    Create 2D plot area

    :param ax: axes of initializer, defaults to new subplot
    :type ax: AxesSubplot, optional
    :param equal: set aspect ratio to 1:1, default False
    :type equal: bool
    :return: initialized axes
    :rtype: AxesSubplot

    Initialize axes with dimensions given by ``dim`` which can be:

        * A (scalar), -A:A x -A:A
        * [A,B], A:B x A:B
        * [A,B,C,D], A:B x C:D

    :seealso: :func:`plotvol3`, :func:`expand_dims`
    """
    dims = expand_dims(dim, 2)
    if ax is None:
        ax = plt.subplot()
    ax.axis(dims)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    if equal:
        ax.set_aspect("equal")
    if grid:
        ax.grid(True)
    return ax


def plotvol3(dim=None, ax=None, equal=True, grid=False, projection="ortho"):
    """
    Create 3D plot volume

    :param ax: axes of initializer, defaults to new subplot
    :type ax: Axes3DSubplot, optional
    :param equal: set aspect ratio to 1:1:1, default False
    :type equal: bool
    :return: initialized axes
    :rtype: Axes3DSubplot

    Initialize axes with dimensions given by ``dim`` which can be:

        * A (scalar), -A:A x -A:A x -A:A
        * [A,B], A:B x A:B x A:B
        * [A,B,C,D,E,F], A:B x C:D x E:F

    :seealso: :func:`plotvol2`, :func:`expand_dims`
    """
    # create an axis if none existing
    ax = axes_logic(ax, 3, projection=projection)

    if dim is None:
        ax.autoscale(True)
    else:
        dims = expand_dims(dim, 3)
        ax.set_xlim3d(dims[0], dims[1])
        ax.set_ylim3d(dims[2], dims[3])
        ax.set_zlim3d(dims[4], dims[5])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    if equal:
        try:
            ax.set_box_aspect((1,) * 3)
        except AttributeError:
            # old version of MPL doesn't support this
            warnings.warn(
                "Current version of matplotlib does not support set_box_aspect()"
            )
    if grid:
        ax.grid(True)
    return ax


def expand_dims(dim=None, nd=2):
    """[summary]

    :param dim: [description], defaults to None
    :type dim: [type], optional
    :param nd: [description], defaults to 2
    :type nd: int, optional
    :raises ValueError: bad arguments
    :return: 2d or 3d dimensions vector
    :rtype: ndarray(4) or ndarray(6)

    Compute bounding dimensions for plots from shorthand notation.

    If ``nd==2``, [xmin, xmax, ymin, ymax]:
        * A -> [-A, A, -A, A]
        * [A,B] -> [A, B, A, B]
        * [A,B,C,D] -> [A, B, C, D]

    If ``nd==3``, [xmin, xmax, ymin, ymax, zmin, zmax]:
        * A -> [-A, A, -A, A, -A, A]
        * [A,B] -> [A, B, A, B, A, B]
        * [A,B,C,D,E,F] -> [A, B, C, D, E, F]
    """
    dim = base.getvector(dim)

    if nd == 2:
        if len(dim) == 1:
            return np.r_[-dim, dim, -dim, dim]
        elif len(dim) == 2:
            return np.r_[-dim[0], dim[0], -dim[1], dim[1]]
        elif len(dim) == 4:
            return dim
        else:
            raise ValueError("bad dimension specified")
    elif nd == 3:
        if len(dim) == 1:
            return np.r_[-dim, dim, -dim, dim, -dim, dim]
        elif len(dim) == 3:
            return np.r_[-dim[0], dim[0], -dim[1], dim[1], -dim[2], dim[2]]
        elif len(dim) == 6:
            return dim
        else:
            raise ValueError("bad dimension specified")
    else:
        raise ValueError("nd is 2 or 3")


def isnotebook():
    """
    Determine if code is being run from a Jupyter notebook

    :references:

        - https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-
          is-executed-in-the-ipython-notebook/39662359#39662359
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


if __name__ == "__main__":
    import pathlib

    exec(
        open(
            pathlib.Path(__file__).parent.parent.parent.absolute()
            / "tests"
            / "base"
            / "test_graphics.py"
        ).read()
    )  # pylint: disable=exec-used
