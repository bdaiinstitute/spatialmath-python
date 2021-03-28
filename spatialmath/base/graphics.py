import matplotlib.pyplot as plt
from numpy.core.defchararray import center
from spatialmath import base
import numpy as np
import scipy as sp

# TODO
# axes_logic everywhere
# dont do draw
# return reference to the graphics object
# don't have own color/style options, go for MPL ones
# unit tests
# seealso
# example code
# return a redrawer object, that can be used for animation


def plot_box(ax=None, 
        bbox=None, bl=None, tl=None, br=None, tr=None, wh=None, centre=None,
        color=None, filled=True, alpha=None, thickness=None, **kwargs):
    """
    Plot a box using matplotlib

    :param ax: the axes to draw on, defaults to ``gca()``
    :type ax: Axis, optional
    :param bbox: bounding box matrix, defaults to None
    :type bbox: ndarray(2,2), optional
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
    :param centre: [description], defaults to None
    :type centre: array_like(2), optional
    :param color: box outline color
    :type color: array_like(3) or str
    :param fillcolor: box fill color
    :type fillcolor: array_like(3) or str
    :param alpha: transparency, defaults to 1
    :type alpha: float, optional
    :param thickness: line thickness, defaults to None
    :type thickness: float, optional
    :return: the matplotlib object
    :rtype: Patch.Rectangle

    Plots a box on the specified axes using matplotlib

    The box can be specified in many ways:

    - bounding box which is a 2x2 matrix [xmin, xmax; ymin, ymax]
    - centre and width+height
    - bottom-left and top-right corners
    - bottom-left corner and width+height
    - top-right corner and width+height
    - top-left corner and width+height
    """

    if bbox is not None:
        xy = bbox[:,0]
        w = bbox[0,1] - bbox[0,0]
        h = bbox[1,1] - bbox[1,0]
    elif bl is not None and tl is None and tr is None and wh is not None and centre is None:
        # bl + wh
        xy = bl
        w, h = wh
    elif bl is not None and tl is None and tr is not None and wh is None and centre is None:
        # bl + tr
        xy = bl
        w = br[0] - bl[0]
        h = br[1] - bl[1]
    elif bl is None and tl is None and tr is None and wh is not None and centre is not None:
        # centre + wh
        w, h = wh
        xy = (centre[0] - w / 2, centre[1] - h / 2)
    elif bl is None and tl is None and tr is not None and wh is not None and centre is None:
        # tr + wh
        w, h = wh
        xy = (tr[0] - wh[0], tr[1] - wh[1])
    elif bl is None and tl is not None and tr is None and wh is not None and centre is None:
        # tl + wh
        w, h = wh
        xy = (tl[0], tl[1] - h)

    ax = _axes_logic(ax, 2)

    if filled:
        r = plt.Rectangle(xy, w, h, edgecolor=color, facecolor=fillcolor, fill=fill,
            alpha=alpha, linewidth=thickness, clip_on=True, **kwargs)
        ax.add_patch(rect)
    else:
        x1 = xy[0]
        x2 = x1 + w
        y1 = xy[1]
        y2 = y1 + h
        r = plt.plot([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1], **kwargs)

    return r


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
    """
    
    defaults = {
        'horizontalalignment': 'left',
        'verticalalignment': 'center'
    }
    for k, v in defaults.items():
        if k not in kwargs:
            kwargs[k] = v
    if ax is None:
        ax = plt.gca()
    plt.text(pos[0], pos[1], text, color=color, **kwargs)


def plot_point(pos, marker='bs', text=None, ax=None, color=None, textargs=None, **kwargs):
    """
    Plot a point using matplotlib

    :param pos: position of marker
    :type pos: array_like(2), ndarray(2,n), list of 2-tuples
    :param marker: matplotlub marker style, defaults to 'bs'
    :type marker: str or list of str, optional
    :param text: text label, defaults to None
    :type text: str, optional
    :param ax: axes to plot in, defaults to ``gca()````
    :type ax: Axis, optional
    :param color: text color, defaults to None
    :type color: str or array_like(3), optional

    The color of the marker can be different to the color of the text,
    the marker color is specified by a single letter in the marker string.

    A point can multiple markers which will be overlaid, for instance ``["rx",
    "ro"]`` will give a ⨂ symbol.

    The optional text label is placed to the right of the marker, and vertically
    aligned. 
    
    Multiple points can be marked if ``pos`` is a 2xn array or a list of
    coordinate pairs.  If a label is provided every point will have the same
    label. However, the text is processed with ``format`` and is provided with a
    single argument, the point index (starting at zero).


    """
    
    if isinstance(pos, np.ndarray):
        if pos.ndim == 1:
            x = pos[0]
            y = pos[1]
        elif pos.ndim == 2 and pos.shape[0] == 2:
            x = pos[0,:]
            y = pos[1,:]
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
        'fontsize': 12, 
        'horizontalalignment': 'left',
        'verticalalignment': 'center'
            }
    if textargs is not None:
        textopts = {**textopts, **textargs}

    if ax is None:
        ax = plt.gca()
    if isinstance(marker, (list, tuple)):
        for m in marker:
            plt.plot(x, y, m, **kwargs)
    else:
        plt.plot(x, y, marker)
    if text:
        try:
            for i, xy in enumerate(zip(x, y)):
                plt.text(xy[0], xy[1], ' ' + text.format(i), color=color, **textopts)
        except:
            plt.text(x, y, ' ' + text, ha='left', va='center', color=color, **textopts)




def _axes_dimensions(ax):
    if hasattr(ax, 'get_zlim'):
        return 3
    else:
        return 2

def circle(centre=(0, 0), radius=1, npoints=50):
    u = np.linspace(0.0, 2.0 * np.pi, npoints)
    x = radius * np.cos(u) + centre[0]
    y = radius * np.sin(u) + centre[1]

    return (x, y)

def plot_circle(centre=(0, 0), radius=1, npoints=50, ax=None, filled=False):

    x, y = circle(centre, radius, npoints)
    ax = _axes_logic(ax, 2)
    if filled:
        patch = plt.Polygon(x, y, **kwargs)
        ax.add_patch(patch)
    else:
        plt.plot(x, y, **kwargs)

def sphere(centre=(0,0,0), radius=1, npoints=50):
    u = np.linspace(0.0, 2.0 * np.pi, npoints)
    v = np.linspace(0.0, np.pi, npoints)

    x = radius * np.outer(np.cos(u), np.sin(v)) + centre[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + centre[1]
    z = radius * np.outer(np.ones_like(u), np.cos(v)) + centre[2]

    return (x, y, z)

def plot_sphere(centre=(0,0,0), radius=1, npoints=50, ax=None, wireframe=False, **kwargs):
    (x, y, z) = _sphere(centre=centre, radius=radius, npoints=npoints)

    ax = _axes_logic(ax, 3)

    if wireframe:
        ax.plot_wireframe(x, y, z, **kwargs)
    else:
        ax.plot_surface(x, y, z, **kwargs)


def ellipse(E, centre=(0,0), scale=1, confidence=None, npoints=40, inverted=False):
    """[summary]

    :param E: ellipse defined by :math:`x^T \mat{E} x = 1`
    :type E: ndarray(2,2)
    :param centre: ellipse centre, defaults to (0,0,0)
    :type centre: tuple, optional
    :param scale:
    :type scale:
    :param confidence: if E is an inverse covariance matrix plot an ellipse
        for this confidence interval in the range [0,1], defaults to None
    :type confidence: float, optional
    :param npoints: number of points on circumferance, defaults to 40
    :type npoints: int, optional
    :param inverted: if :math:`\mat{E}^{-1}` is provided, defaults to False
    :type inverted: bool, optional
    :raises ValueError: [description]
    :return: x and y coordinates
    :rtype: tuple of ndarray(1)

    .. note:: In some problems we compute :math:`\mat{E}^{-1}` so to avoid
        inverting ``E`` twice to compute the ellipse, we flag that the inverse
        is provided using ``inverted``.  For example: 
        
        - for robot manipulability
        :math:`\nu (\mat{J} \mat{J}^T)^{-1} \nu` i 
        - a covariance matrix
        :math:`(x - \mu)^T \mat{P}^{-1} (x - \mu)`
    """
    if E.shape != (2,2):
        raise ValueError('ellipse is defined by a 2x2 matrix')

    if confidence:
        # process the probability
        s = sqrt(chi2inv(confidence, 2)) * scale
    else:
        s = scale

    x, y = circle()  # unit circle

    if not inverted:
        E = np.linalg.inv(E)

    e = s * sp.linalg.sqrtm(E) @ np.array([x, y]) + np.c_[centre]
    return e[0,:], e[1,:]

def plot_ellipse(E, centre=(0,0), scale=1, confidence=None, npoints=40, inverted=False, ax=None, filled=None, **kwargs):
    
    # allow for centre[2] to plot ellipse in a plane in a 3D plot

    x, y = ellipse(E, centre, scale, confidence, npoints, inverted)
    ax = _axes_logic(ax, 2)
    if filled:
        patch = plt.Polygon(x, y, **kwargs)
        ax.add_patch(patch)
    else:
        plt.plot(x, y, **kwargs)

def ellipsoid(E, centre=(0,0,0), scale=1, confidence=None, npoints=40, inverted=False):

    if E.shape != (3,3):
        raise ValueError('ellipsoid is defined by a 3x3 matrix')

    if confidence:
        # process the probability
        from scipy.stats.distributions import chi2
        s = math.sqrt(chi2.ppf(s, df=2)) * scale
    else:
        s = scale

    if not inverted:
        E = np.linalg.inv(E)

    x, y, z = sphere()  # unit sphere
    e = s * sp.linalg.sqrtm(E) @ np.array([x.flatten(), y.flatten(), z.flatten()]) + np.c_[centre].T
    return e[0,:].reshape(x.shape), e[1,:].reshape(x.shape), e[2,:].reshape(x.shape)

def plot_ellipsoid(E, centre=(0,0,0), scale=1, confidence=None, npoints=40, inverted=False, ax=None, wireframe=False, stride=1, **kwargs):
    """
    Draw an ellipsoid

    :param E: ellipsoid
    :type E: ndarray(3,3)
    :param centre: [description], defaults to (0,0,0)
    :type centre: tuple, optional
        :param scale:
    :type scale:
    :param confidence: confidence interval, range 0 to 1
    :type confidence: float
    :param npoints: [description], defaults to 40
    :type npoints: int, optional
    :param inverted: [description], defaults to False
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
    """
    x, y, z = ellipsoid(E, centre, scale, confidence, npoints, inverted)
    ax = _axes_logic(ax, 3)
    if wireframe:
        return ax.plot_wireframe(x, y, z, rstride=stride, cstride=stride, **kwargs)
    else:
        return ax.plot_surface(x, y, z, **kwargs)

def _axes_logic(ax, dimensions, projection='ortho'):
    if ax is not None:
        # axis was given
        if _axes_dimensions == dimensions:
            return ax
        # mismatch, create new axes
    
    # no axis specified
    if dimensions == 2:
        ax = plt.axes()
    else:
        ax = plt.axes(projection='3d', proj_type=projection)
    return ax

def isnotebook():
    """
    Determine if code is being run from a Jupyter notebook

    ``_isnotebook`` is True if running Jupyter notebook, else False

    :references:

        - https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-
          is-executed-in-the-ipython-notebook/39662359#39662359
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

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
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    if equal:
        ax.set_aspect('equal')
    if grid:
        ax.grid(True)
    return ax

def plotvol3(dim, ax=None, equal=True, grid=False, projection='ortho'):
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
    dims = expand_dims(dim, 3)
    if ax is None:
        ax = plt.subplot(projection='3d', proj_type=projection)
    ax.set_xlim3d(dims[0], dims[1])
    ax.set_ylim3d(dims[2], dims[3])
    ax.set_zlim3d(dims[4], dims[5])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if equal:
        ax.set_box_aspect((1,) * 3)
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
            raise ValueError('bad dimension specified')
    elif nd == 3:
        if len(dim) == 1:
                return np.r_[-dim, dim, -dim, dim, -dim, dim]
        elif len(dim) == 3:
                return np.r_[-dim[0], dim[0], -dim[1], dim[1], -dim[2], dim[2]]
        elif len(dim) == 6:
                return dim
        else:
            raise ValueError('bad dimension specified')
    else:
        raise ValueError('nd is 2 or 3')
