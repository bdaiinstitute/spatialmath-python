import math
from itertools import product
import warnings
import numpy as np
import scipy as sp
from matplotlib import colors

from spatialmath import base as smbase

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from mpl_toolkits.mplot3d.art3d import (
        Poly3DCollection,
        Line3DCollection,
        pathpatch_2d_to_3d,
    )

    _matplotlib_exists = True
except ImportError:  # pragma: no cover
    _matplotlib_exists = False

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

    handle = ax.text(pos[0], pos[1], text, color=color, **kwargs)
    return [handle]


def plot_point(pos, marker="bs", text=None, ax=None, textargs=None, textcolor=None, **kwargs):
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
      coordinate pairs.  In this case:
        - all points have the same ``text`` label
        - ``text`` can include the format string {} which is susbstituted for the
          point index, starting at zero
        - ``text`` can be a tuple containing a format string followed by vectors
          of shape(n).  For example::

            ``("#{0} a={1:.1f}, b={2:.1f}", a, b)``

          will label each point with its index (argument 0) and consecutive
          elements of ``a`` and ``b`` which are arguments 1 and 2 respectively.

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
        if smbase.islistof(pos, (tuple, list)):
            x = [z[0] for z in pos]
            y = [z[1] for z in pos]
        elif smbase.islistof(pos, np.ndarray):
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
    if textcolor is not None and "color" not in textopts:
        textopts["color"] = textcolor

    if ax is None:
        ax = plt.gca()

    handles = []
    if isinstance(marker, (list, tuple)):
        for m in marker:
            handles.append(ax.plot(x, y, m, **kwargs))
    else:
        handles.append(ax.plot(x, y, marker, **kwargs))
    if text is not None:
        try:
            xy = zip(x, y)
        except TypeError:
            xy = [(x, y)]
        if isinstance(text, str):
            # simple string, but might have format chars
            for i, (x, y) in enumerate(xy):
                handles.append(ax.text(x, y, " " + text.format(i), **textopts))
        elif isinstance(text, (tuple, list)):
            for i, (x, y) in enumerate(xy):
                handles.append(
                    ax.text(
                        x,
                        y,
                        " " + text[0].format(i, *[d[i] for d in text[1:]]),
                        **textopts
                    )
                )
    return handles


def plot_homline(lines, *args, ax=None, xlim=None, ylim=None, **kwargs):
    r"""
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
    if xlim is None:
        xlim = np.r_[ax.get_xlim()]
    if ylim is None:
        ylim = np.r_[ax.get_ylim()]

    # if lines.ndim == 1:
    #     lines = lines.
    lines = smbase.getmatrix(lines, (3, None))

    handles = []
    for line in lines.T:  # for each column
        if abs(line[1]) > abs(line[0]):
            y = (-line[2] - line[0] * xlim) / line[1]
            ax.plot(xlim, y, *args, **kwargs)
        else:
            x = (-line[2] - line[1] * ylim) / line[0]
            handles.append(ax.plot(x, ylim, *args, **kwargs))

    return handles


def plot_box(
    *fmt,
    lbrt=None,
    lrbt=None,
    lbwh=None,
    bbox=None,
    ltrb=None,
    lb=None,
    lt=None,
    rb=None,
    rt=None,
    wh=None,
    centre=None,
    w=None,
    h=None,
    ax=None,
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
    :param tr: top-right corner, defaults to None
    :type tr: array_like(2), optional
    :param wh: width and height, if both are the same provide scalar, defaults to None
    :type wh: scalar, array_like(2), optional
    :param centre: centre of box, defaults to None
    :type centre: array_like(2), optional
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

    - bounding box which is a 2x2 matrix [xmin, xmax, ymin, ymax]
    - bounding box [xmin, xmax, ymin, ymax]
    - alternative box [xmin, ymin, xmax, ymax]
    - centre and width+height
    - bottom-left and top-right corners
    - bottom-left corner and width+height
    - top-right corner and width+height
    - top-left corner and width+height

    For plots where the y-axis is inverted (eg. for images) then top is the
    smaller vertical coordinate.

    Example:

    .. runblock:: pycon

        >>> from spatialmath.base import plotvol2, plot_box
        >>> plotvol2(5)
        >>> plot_box('r', centre=(2,3), wh=1) # w=h=1
        >>> plot_box(tl=(1,1), br=(0,2), filled=True, color='b')
    """

    if wh is not None:
        if smbase.isscalar(wh):
            w, h = wh, wh
        else:
            w, h = wh
    
    # test for various 4-coordinate versions
    if bbox is not None:
        lb = bbox[:2]
        w, h = bbox[2:]

    elif lbwh is not None:
        lb = lbwh[:2]
        w, h = lbwh[2:]

    elif lbrt is not None:
        lb = lbrt[:2]
        rt = lbrt[2:]
        w, h = rt[0] - lb[0], rt[1] - lb[1]

    elif lrbt is not None:
        lb = (lrbt[0], lrbt[2])
        rt = (lrbt[1], lrbt[3])
        w, h = rt[0] - lb[0], rt[1] - lb[1]

    elif ltrb is not None:
        lb = (ltrb[0], ltrb[3])
        rt = (ltrb[2], ltrb[1])
        w, h = rt[0] - lb[0], rt[1] - lb[1]

    elif w is not None and h is not None:
        # we have width & height, one corner is enough

        if centre is not None:
            lb = (centre[0] - w/2, centre[1] - h/2)

        elif lt is not None:
            lb = (lt[0], lt[1] - h)

        elif rt is not None:
            lb = (rt[0] - w, rt[1] - h)

        elif rb is not None:
            lb = (rb[0] - w, rb[1])

    else:
        # we need two opposite corners
        if lb is not None and rt is not None:
            w = rt[0] - lb[0]
            h = rt[1] - lb[1]

        elif lt is not None and rb is not None:
            lb = (lt[0], rb[1])
            w = rb[0] - lt[0]
            h = lt[1] - rb[1]

        else:
            raise ValueError('cant compute box')

    if w < 0:
        raise ValueError("width must be positive")
    if h < 0:
        raise ValueError("height must be positive")

    # we only need lb, wh
    ax = axes_logic(ax, 2)
    if filled:
        r = plt.Rectangle(lb, w, h, clip_on=True, **kwargs)
    else:
        if 'color' in kwargs:
            kwargs['edgecolor'] = kwargs['color']
            del kwargs['color']
        r = plt.Rectangle(lb, w, h, clip_on=True, facecolor='None', **kwargs)
    ax.add_patch(r)

    return r

def plot_arrow(start, end, ax=None, **kwargs):
    """
    Plot 2D arrow

    :param start: start point, arrow tail
    :type start: array_like(2)
    :param end: end point, arrow head
    :type end: array_like(2)
    :param ax: axes to draw into, defaults to None
    :type ax: Axes, optional
    :param kwargs: argumetns to pass to :class:`matplotlib.patches.Arrow`

    Example:

    .. runblock:: pycon

        >>> from spatialmath.base import plotvol2, plot_arrow
        >>> plotvol2(5)
        >>> plot_arrow((-2, 2), (3, 4), color='r', width=0.1)  # red arrow
    """
    ax = axes_logic(ax, 2)

    ax.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1], length_includes_head=True, **kwargs)

def plot_polygon(vertices, *fmt, close=False, **kwargs):
    """
    Plot polygon

    :param vertices: vertices
    :type vertices: ndarray(2,N)
    :param close: close the polygon, defaults to False
    :type close: bool, optional
    :param kwargs: arguments passed to Patch
    :return: Matplotlib artist
    :rtype: line or patch

    Example:

    .. runblock:: pycon

        >>> from spatialmath.base import plotvol2, plot_polygon
        >>> plotvol2(5)
        >>> vertices = np.array([[-1, 2, -1], [1, 0, -1]])
        >>> plot_polygon(vertices, filled=True, facecolor='g')  # green filled triangle
    """

    if close:
        vertices = np.hstack((vertices, vertices[:, [0]]))
    return _render2D(vertices, fmt=fmt, **kwargs)


def _render2D(vertices, pose=None, filled=False, color=None, ax=None, fmt=(), **kwargs):

    ax = axes_logic(ax, 2)
    if pose is not None:
        vertices = pose * vertices

    if filled:
        if color is not None:
            kwargs['facecolor'] = color
            kwargs['edgecolor'] = color
        r = plt.Polygon(vertices.T, closed=True, **kwargs)
        ax.add_patch(r)
    else:
        r = plt.plot(vertices[0, :], vertices[1, :], *fmt, color=color, **kwargs)
    return r


def circle(centre=(0, 0), radius=1, resolution=50, closed=False):
    """
    Points on a circle

    :param centre: centre of circle, defaults to (0, 0)
    :type centre: array_like(2), optional
    :param radius: radius of circle, defaults to 1
    :type radius: float, optional
    :param resolution: number of points on circumferece, defaults to 50
    :type resolution: int, optional
    :return: points on circumference
    :rtype: ndarray(2,N) or ndarray(3,N)

    Returns a set of ``resolution`` that lie on the circumference of a circle
    of given ``center`` and ``radius``.

    If ``len(centre)==3`` then the 3D coordinates are returned, where the
    circle lies in the xy-plane and the z-coordinate comes from ``centre[2]``.
    """
    if closed:
        resolution += 1
    u = np.linspace(0.0, 2.0 * np.pi, resolution, endpoint=closed)
    x = radius * np.cos(u) + centre[0]
    y = radius * np.sin(u) + centre[1]
    if len(centre) == 3:
        z = np.full(x.shape, centre[2])
        return np.array((x, y, z))
    else:
        return np.array((x, y))


def plot_circle(
    radius, centre, *fmt, resolution=50, ax=None, filled=False, **kwargs
):
    """
    Plot a circle using matplotlib

    :param centre: centre of circle, defaults to (0,0)
    :type centre: array_like(2), optional
    :param args:
    :param radius: radius of circle
    :type radius: float
    :param resolution: number of points on circumference, defaults to 50
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
    centres = smbase.getmatrix(centre, (2, None))

    ax = axes_logic(ax, 2)
    handles = []
    for centre in centres.T:
        xy = circle(centre, radius, resolution, closed=not filled)
        if filled:
            patch = plt.Polygon(xy.T, **kwargs)
            handles.append(ax.add_patch(patch))
        else:
            handles.append(ax.plot(xy[0, :], xy[1, :], *fmt, **kwargs))
    return handles


def ellipse(E, centre=(0, 0), scale=1, confidence=None, resolution=40, inverted=False, closed=False):
    r"""
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
        from scipy.stats.distributions import chi2

        # process the probability
        s = math.sqrt(chi2.ppf(confidence, df=2)) * scale
    else:
        s = scale

    xy = circle(resolution=resolution, closed=closed)  # unit circle

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
    filled=False,
    **kwargs
):
    r"""
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

    xy = ellipse(E, centre, scale, confidence, resolution, inverted, closed=True)
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
    theta_range = np.linspace(0, np.pi, resolution)
    phi_range = np.linspace(-np.pi, np.pi, resolution)

    Phi, Theta = np.meshgrid(phi_range, theta_range)

    x = radius * np.sin(Theta) * np.cos(Phi) + centre[0]
    y = radius * np.sin(Theta) * np.sin(Phi) + centre[1]
    z = radius * np.cos(Theta) + centre[2]

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
        >>> plot_sphere(radius=1, color='r')   # red sphere wireframe
        >>> plot_sphere(radius=1, centre=(1,1,1), filled=True, facecolor='b')

    :seealso: :func:`~matplotlib.pyplot.plot_surface`, :func:`~matplotlib.pyplot.plot_wireframe`
    """
    ax = axes_logic(ax, 3)

    centre = smbase.getmatrix(centre, (3, None))

    handles = []
    for c in centre.T:
        X, Y, Z = sphere(centre=c, radius=radius, resolution=resolution)
        handles.append(_render3D(ax, X, Y, Z, **kwargs))

    return handles


def ellipsoid(
    E, centre=(0, 0, 0), scale=1, confidence=None, resolution=40, inverted=False
):
    r"""
    rPoints on an ellipsoid

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

        s = math.sqrt(chi2.ppf(confidence, df=3)) * scale
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
    r"""
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

    Example::

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

# TODO, get cylinder, cuboid, cone working
def cylinder(center_x, center_y, radius, height_z, resolution=50):
    Z = np.linspace(0, height_z, radius)
    theta = np.linspace(0, 2 * np.pi, radius)
    theta_grid, z_grid = np.meshgrid(theta, z)
    X = radius * np.cos(theta_grid) + center_x
    Y = radius * np.sin(theta_grid) + center_y
    return X, Y, Z

# https://stackoverflow.com/questions/30715083/python-plotting-a-wireframe-3d-cuboid
# https://stackoverflow.com/questions/26874791/disconnected-surfaces-when-plotting-cones
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

    :param radius: radius of sphere
    :type radius: float
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
    if smbase.isscalar(height):
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


def plot_cone(
    radius,
    height,
    resolution=50,
    flip=False,
    centre=(0, 0, 0),
    ends=False,
    ax=None,
    filled=False,
    **kwargs
):
    """
    Plot a cone using matplotlib

    :param radius: radius of cone at open end
    :type radius: float
    :param height: height of cone in the z-direction
    :type height: float
    :param resolution: number of points on circumferece, defaults to 50
    :type resolution: int, optional
    :param flip: cone faces upward, defaults to False
    :type flip: bool, optional

    :param pose: pose of cone, defaults to None
    :type pose: SE3, optional
    :param ax: axes to draw into, defaults to None
    :type ax: Axes3D, optional
    :param filled: draw filled polygon, else wireframe, defaults to False
    :type filled: bool, optional
    :param kwargs: arguments passed to ``plot_wireframe`` or ``plot_surface``

    :return: matplotlib objects
    :rtype: list of matplotlib object types

    The axis of the cone is parallel to the z-axis and it is drawn pointing
    down. The point is at z=0 and the open end at z= ``height``.  If ``flip`` is
    True then the cone faces upwards, the point is at z= ``height`` and the open
    end at z=0.

    The cylinder can be positioned by setting ``centre``, or positioned
    and orientated by setting ``pose``.

    :seealso: :func:`~matplotlib.pyplot.plot_surface`, :func:`~matplotlib.pyplot.plot_wireframe`
    """
    ax = axes_logic(ax, 3)

    # https://stackoverflow.com/questions/26874791/disconnected-surfaces-when-plotting-cones
    # Set up the grid in polar coords
    theta = np.linspace(0, 2 * np.pi, resolution)
    r = np.linspace(0, radius, resolution)
    T, R = np.meshgrid(theta, r)

    # Then calculate X, Y, and Z
    X = R * np.cos(T) + centre[0]
    Y = R * np.sin(T) + centre[1]
    Z = np.sqrt(X ** 2 + Y ** 2) / radius * height + centre[2]
    if flip:
        Z = height - Z

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
        vertices = smbase.homtrans(pose.A, vertices)

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
        if 'color' in kwargs:
            if 'alpha' in kwargs:
                alpha = kwargs['alpha']
                del kwargs['alpha']
            else:
                alpha = 1
            kwargs['colors'] = colors.to_rgba(kwargs['color'], alpha)
            del kwargs['color']
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
        return ax.plot_surface(X, Y, Z, color=color, **kwargs)
    else:
        kwargs["colors"] = color
        return ax.plot_wireframe(X, Y, Z, **kwargs)


def _axes_dimensions(ax):
    """
    Dimensions of axes

    :param ax: axes
    :type ax: Axes3DSubplot or AxesSubplot
    :return: dimensionality of axes, either 2 or 3
    :rtype: int
    """
    classname = ax.__class__.__name__

    if classname in ("Axes3DSubplot", "Animate"):
        return 3
    elif classname in ("AxesSubplot", "Animate2"):
        return 2


def axes_get_limits(ax):
    return np.r_[ax.get_xlim(), ax.get_ylim()]


def axes_get_scale(ax):
    limits = axes_get_limits(ax)
    return max(abs(limits[1] - limits[0]), abs(limits[3] - limits[2]))


def axes_logic(ax, dimensions, projection="ortho", autoscale=True):
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

    if not _matplotlib_exists:
        raise NotImplementedError("Matplotlib is not installed: pip install matplotlib")

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
            # print("use existing axes")
            return ax
        # mismatch in dimensions, create new axes
    # print('create new axes')
    plt.figure()
    # no axis specified
    if dimensions == 2:
        ax = plt.axes()
        if autoscale:
            ax.autoscale()
    else:
        ax = plt.axes(projection="3d", proj_type=projection)

    plt.sca(ax)
    plt.axes(ax)

    return ax


def plotvol2(dim, ax=None, equal=True, grid=False, labels=True):
    """
    Create 2D plot area

    :param ax: axes of initializer, defaults to new subplot
    :type ax: AxesSubplot, optional
    :param equal: set aspect ratio to 1:1, default False
    :type equal: bool
    :return: initialized axes
    :rtype: AxesSubplot

    Initialize axes with dimensions given by ``dim`` which can be:

    ==============  ======  ======
    input           xrange  yrange
    ==============  ======  ======
    A (scalar)      -A:A    -A:A
    [A, B]           A:B     A:B
    [A, B, C, D]     A:B     C:D
    ==============  ======  ======

    :seealso: :func:`plotvol3`, :func:`expand_dims`
    """
    dims = expand_dims(dim, 2)
    if ax is None:
        ax = plt.subplot()
    ax.axis(dims)
    if labels:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    if equal:
        ax.set_aspect("equal")
    if grid:
        ax.grid(True)
        ax.set_axisbelow(True)

    # signal to related functions that plotvol set the axis limits
    ax._plotvol = True
    return ax


def plotvol3(
    dim=None, ax=None, equal=True, grid=False, labels=True, projection="ortho"
):
    """
    Create 3D plot volume

    :param ax: axes of initializer, defaults to new subplot
    :type ax: Axes3DSubplot, optional
    :param equal: set aspect ratio to 1:1:1, default False
    :type equal: bool
    :return: initialized axes
    :rtype: Axes3DSubplot

    Initialize axes with dimensions given by ``dim`` which can be:

    ==================  ======  ======  =======
    input               xrange  yrange  zrange
    ==================  ======  ======  =======
    A (scalar)          -A:A    -A:A    -A:A
    [A, B]              A:B     A:B     A:B
    [A, B, C, D, E, F]  A:B     C:D     E:F
    ==================  ======  ======  =======

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
        if labels:
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

    # signal to related functions that plotvol set the axis limits
    ax._plotvol = True
    return ax


def expand_dims(dim=None, nd=2):
    """
    Expand compact axis dimensions

    :param dim: dimensions, defaults to None
    :type dim: scalar, array_like(2), array_like(4), array_like(6), optional
    :param nd: number of axes dimensions, defaults to 2
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
    dim = smbase.getvector(dim)

    if nd == 2:
        if len(dim) == 1:
            return np.r_[-dim, dim, -dim, dim]
        elif len(dim) == 2:
            return np.r_[dim[0], dim[1], dim[0], dim[1]]
        elif len(dim) == 4:
            return dim
        else:
            raise ValueError("bad dimension specified")
    elif nd == 3:
        if len(dim) == 1:
            return np.r_[-dim, dim, -dim, dim, -dim, dim]
        elif len(dim) == 2:
            return np.r_[dim[0], dim[1], dim[0], dim[1], dim[0], dim[1]]
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


    plotvol2(5)
    # plot_box(ltrb=[-1, 4, 2, 2], color='r', linewidth=2)
    # plot_box(lbrt=[-1, 2, 2, 4], color='k', linestyle='--', linewidth=4)
    # plot_box(lbwh=[2, -2, 2, 3], color='k', linewidth=2)
    # plot_box(centre=(-2, -1), wh=2, color='b', linewidth=2)
    # plot_box(centre=(-2, -1), wh=(1,3), color='g', linewidth=2)
    # plt.grid(True)
    # plt.show(block=True)

    # plt.imshow(np.eye(200))
    # umin, umax, vmin, vmax = 23, 166, 110, 212
    # plot_box(l=umin, r=umax, t=vmin, b=vmax, color="g")

    plot_circle(1, (2,3), resolution=3, filled=False)
    plt.show(block=True)
    
    exec(
        open(
            pathlib.Path(__file__).parent.parent.parent.absolute()
            / "tests"
            / "base"
            / "test_graphics.py"
        ).read()
    )  # pylint: disable=exec-used
