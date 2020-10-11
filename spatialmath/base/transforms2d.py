"""
This modules contains functions to create and transform rotation matrices
and homogeneous tranformation matrices.

Vector arguments are what numpy refers to as ``array_like`` and can be a list,
tuple, numpy array, numpy row vector or numpy column vector.

"""

# This file is part of the SpatialMath toolbox for Python
# https://github.com/petercorke/spatialmath-python
#
# MIT License
#
# Copyright (c) 1993-2020 Peter Corke
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Contributors:
#
#     1. Luis Fernando Lara Tobar and Peter Corke, 2008
#     2. Josh Carrigg Hodson, Aditya Dua, Chee Ho Chan, 2017 (robopy)
#     3. Peter Corke, 2020

# pylint: disable=invalid-name

import sys
import math
import numpy as np
import scipy.linalg
from spatialmath.base import argcheck
from spatialmath.base import vectors as vec
from spatialmath.base import transformsNd as trn
from spatialmath.base import animate
from spatialmath.base import symbolic as sym

_eps = np.finfo(np.float64).eps


# ---------------------------------------------------------------------------------------#
def rot2(theta, unit='rad'):
    """
    Create SO(2) rotation

    :param theta: rotation angle
    :type theta: float
    :param unit: angular units: 'rad' [default], or 'deg'
    :type unit: str
    :return: 2x2 rotation matrix
    :rtype: numpy.ndarray, shape=(2,2)

    - ``ROT2(THETA)`` is an SO(2) rotation matrix (2x2) representing a rotation of THETA radians.
    - ``ROT2(THETA, 'deg')`` as above but THETA is in degrees.
    """
    theta = argcheck.getunit(theta, unit)
    ct = sym.cos(theta)
    st = sym.sin(theta)
    R = np.array([
        [ct, -st],
        [st, ct]])
    return R


# ---------------------------------------------------------------------------------------#
def trot2(theta, unit='rad', t=None):
    """
    Create SE(2) pure rotation

    :param theta: rotation angle about X-axis
    :type theta: float
    :param unit: angular units: 'rad' [default], or 'deg'
    :type unit: str
    :param t: translation 2-vector, defaults to [0,0]
    :type t: array_like    :return: 3x3 homogeneous transformation matrix
    :rtype: numpy.ndarray, shape=(3,3)

    - ``TROT2(THETA)`` is a homogeneous transformation (3x3) representing a rotation of
      THETA radians.
    - ``TROT2(THETA, 'deg')`` as above but THETA is in degrees.

    Notes:
    - Translational component is zero.
    """
    T = np.pad(rot2(theta, unit), (0, 1), mode='constant')
    if t is not None:
        T[:2, 2] = argcheck.getvector(t, 2, 'array')
    T[2, 2] = 1.0
    return T


# ---------------------------------------------------------------------------------------#
def transl2(x, y=None):
    """
    Create SE(2) pure translation, or extract translation from SE(2) matrix

    :param x: translation along X-axis
    :type x: float
    :param y: translation along Y-axis
    :type y: float
    :return: homogeneous transform matrix or the translation elements of a homogeneous transform
    :rtype: numpy.ndarray, shape=(3,3)

    Create a translational SE(2) matrix:

    - ``T = transl2([X, Y])`` is an SE(2) homogeneous transform (3x3) representing a
      pure translation.
    - ``T = transl2( V )`` as above but the translation is given by a 2-element
      list, dict, or a numpy array, row or column vector.


    Extract the translational part of an SE(2) matrix:

    P = TRANSL2(T) is the translational part of a homogeneous transform as a
    2-element numpy array.
    """

    if np.isscalar(x):
        T = np.identity(3)
        T[:2, 2] = [x, y]
        return T
    elif argcheck.isvector(x, 2):
        T = np.identity(3)
        T[:2, 2] = argcheck.getvector(x, 2)
        return T
    elif argcheck.ismatrix(x, (3, 3)):
        return x[:2, 2]
    else:
        ValueError('bad argument')


def ishom2(T, check=False):
    """
    Test if matrix belongs to SE(2)

    :param T: matrix to test
    :type T: numpy.ndarray
    :param check: check validity of rotation submatrix
    :type check: bool
    :return: whether matrix is an SE(2) homogeneous transformation matrix
    :rtype: bool

    - ``ISHOM2(T)`` is True if the argument ``T`` is of dimension 3x3
    - ``ISHOM2(T, check=True)`` as above, but also checks orthogonality of the
      rotation sub-matrix and validitity of the bottom row.

    :seealso: isR, isrot2, ishom, isvec
    """
    return isinstance(T, np.ndarray) and T.shape == (3, 3) \
        and (not check or (trn.isR(T[:2, :2])
                           and np.all(T[2, :] == np.array([0, 0, 1]))))


def isrot2(R, check=False):
    """
    Test if matrix belongs to SO(2)

    :param R: matrix to test
    :type R: numpy.ndarray
    :param check: check validity of rotation submatrix
    :type check: bool
    :return: whether matrix is an SO(2) rotation matrix
    :rtype: bool

    - ``ISROT(R)`` is True if the argument ``R`` is of dimension 2x2
    - ``ISROT(R, check=True)`` as above, but also checks orthogonality of the rotation matrix.

    :seealso: isR, ishom2, isrot
    """
    return isinstance(R, np.ndarray) and R.shape == (2, 2) \
        and (not check or trn.isR(R))

# ---------------------------------------------------------------------------------------#


def trlog2(T, check=True, twist=False):
    """
    Logarithm of SO(2) or SE(2) matrix

    :param T: SO(2) or SE(2) matrix
    :type T: numpy.ndarray, shape=(2,2) or (3,3)
    :param check: check that matrix is valid
    :type check: bool
    :param twist: return a twist vector instead of matrix [default]
    :type twist: bool
    :return: logarithm
    :rtype: numpy.ndarray, shape=(2,2) or (3,3)
    :raises: ValueError

    An efficient closed-form solution of the matrix logarithm for arguments that
    are SO(2) or SE(2).

    - ``trlog2(R)`` is the logarithm of the passed rotation matrix ``R`` which
      will be 2x2 skew-symmetric matrix.  The equivalent vector from ``vex()``
      is parallel to rotation axis and its norm is the amount of rotation about
      that axis.
    - ``trlog(T)`` is the logarithm of the passed homogeneous transformation
      matrix ``T`` which will be 3x3 augumented skew-symmetric matrix. The
      equivalent vector from ``vexa()`` is the twist vector (6x1) comprising [v
      w].

    :seealso: :func:`~trexp`, :func:`~spatialmath.base.transformsNd.vex`,
              :func:`~spatialmath.base.transformsNd.vexa`
    """

    if ishom2(T, check=check):
        # SE(2) matrix

        if trn.iseye(T):
            # is identity matrix
            if twist:
                return np.zeros((3,))
            else:
                return np.zeros((3, 3))
        else:
            if twist:
                return trn.vexa(scipy.linalg.logm(T))
            else:
                return scipy.linalg.logm(T)

    elif isrot2(T, check=check):
        # SO(2) rotation matrix
        if twist:
            return trn.vex(scipy.linalg.logm(T))
        else:
            return scipy.linalg.logm(T)
    else:
        raise ValueError("Expect SO(2) or SE(2) matrix")
# ---------------------------------------------------------------------------------------#


def trexp2(S, theta=None, check=True):
    """
    Exponential of so(2) or se(2) matrix

    :param S: so(2), se(2) matrix or equivalent velctor
    :type T: numpy.ndarray, shape=(2,2) or (3,3); array_like
    :param theta: motion
    :type theta: float
    :return: 2x2 or 3x3 matrix exponential in SO(2) or SE(2)
    :rtype: numpy.ndarray, shape=(2,2) or (3,3)

    An efficient closed-form solution of the matrix exponential for arguments
    that are so(2) or se(2).

    For so(2) the results is an SO(2) rotation matrix:

    - ``trexp2(S)`` is the matrix exponential of the so(3) element ``S`` which is a 2x2
      skew-symmetric matrix.
    - ``trexp2(S, THETA)`` as above but for an so(3) motion of S*THETA, where ``S`` is
      unit-norm skew-symmetric matrix representing a rotation axis and a rotation magnitude
      given by ``THETA``.
    - ``trexp2(W)`` is the matrix exponential of the so(2) element ``W`` expressed as
      a 1-vector (array_like).
    - ``trexp2(W, THETA)`` as above but for an so(3) motion of W*THETA where ``W`` is a
      unit-norm vector representing a rotation axis and a rotation magnitude
      given by ``THETA``. ``W`` is expressed as a 1-vector (array_like).


    For se(2) the results is an SE(2) homogeneous transformation matrix:

    - ``trexp2(SIGMA)`` is the matrix exponential of the se(2) element ``SIGMA`` which is
      a 3x3 augmented skew-symmetric matrix.
    - ``trexp2(SIGMA, THETA)`` as above but for an se(3) motion of SIGMA*THETA, where ``SIGMA``
      must represent a unit-twist, ie. the rotational component is a unit-norm skew-symmetric
      matrix.
    - ``trexp2(TW)`` is the matrix exponential of the se(3) element ``TW`` represented as
      a 3-vector which can be considered a screw motion.
    - ``trexp2(TW, THETA)`` as above but for an se(2) motion of TW*THETA, where ``TW``
      must represent a unit-twist, ie. the rotational component is a unit-norm skew-symmetric
      matrix.

     :seealso: trlog, trexp2
    """

    if argcheck.ismatrix(S, (3, 3)) or argcheck.isvector(S, 3):
        # se(2) case
        if argcheck.ismatrix(S, (3, 3)):
            # augmentented skew matrix
            if check:
                assert trn.isskewa(S), 'argument must be a valid se(2) element'
            tw = trn.vexa(S)
        else:
            # 3 vector
            tw = argcheck.getvector(S)

        if vec.iszerovec(tw):
            return np.eye(3)

        if theta is None:
            (tw, theta) = vec.unittwist2(tw)
        else:
            assert vec.isunittwist2(tw), 'If theta is specified S must be a unit twist'

        t = tw[0:2]
        w = tw[2]

        R = trn.rodrigues(w, theta)

        skw = trn.skew(w)
        V = np.eye(2) * theta + (1.0 - math.cos(theta)) * skw + (theta - math.sin(theta)) * skw @ skw

        return trn.rt2tr(R, V@t)

    elif argcheck.ismatrix(S, (2, 2)) or argcheck.isvector(S, 1):
        # so(2) case
        if argcheck.ismatrix(S, (2, 2)):
            # skew symmetric matrix
            if check:
                assert trn.isskew(S), 'argument must be a valid so(2) element'
            w = trn.vex(S)
        else:
            # 1 vector
            w = argcheck.getvector(S)

        if theta is not None:
            assert vec.isunitvec(w), 'If theta is specified S must be a unit twist'

        # do Rodrigues' formula for rotation
        return trn.rodrigues(w, theta)
    else:
        raise ValueError(" First argument must be SO(2), 1-vector, SE(2) or 3-vector")


def trinterp2(start, end, s=None):
    """
    Interpolate SE(2) matrices

    :param start: initial SO(2) or SE(2) matrix value when s=0, if None then identity is used
    :type T1: np.ndarray, shape=(2,2), (3,3) or None
    :param end: final SO(2) or SE(2) matrix, value when s=1
    :type T0: np.ndarray, shape=(2,2), (3,3)
    :param s: interpolation coefficient, range 0 to 1
    :type s: float
    :return: SO(2) or SE(2) matrix
    :rtype: np.ndarray, shape=(2,2), (3,3)

    - ``trinterp2(None, T, S)`` is a homogeneous transform (3x3) interpolated
      between identity when S=0 and T (3x3) when S=1.
    - ``trinterp2(T0, T1, S)`` as above but interpolated
      between T0 (3x3) when S=0 and T1 (3x3) when S=1.
    - ``trinterp2(None, R, S)`` is a rotation matrix (2x2) interpolated
      between identity when S=0 and R (2x2) when S=1.
    - ``trinterp2(R0, R1, S)`` as above but interpolated
      between R0 (2x2) when S=0 and R1 (2x2) when S=1.

    Notes:

    - Rotation angle is linearly interpolated.

    :seealso: :func:`~spatialmath.base.transforms3d.trinterp`

    %## 2d homogeneous trajectory
    """
    if argcheck.ismatrix(end, (2, 2)):
        # SO(2) case
        if start is None:
            #	TRINTERP2(T, s)

            th0 = math.atan2(end[1, 0], end[0, 0])

            th = s * th0
        else:
            #	TRINTERP2(T1, start= s)
            assert start.shape == end.shape, 'start and end matrices must be same shape'

            th0 = math.atan2(start[1, 0], start[0, 0])
            th1 = math.atan2(end[1, 0], end[0, 0])

            th = th0 * (1 - s) + s * th1

        return rot2(th)
    elif argcheck.ismatrix(end, (3, 3)):
        if start is None:
            #	TRINTERP2(T, s)

            th0 = math.atan2(end[1, 0], end[0, 0])
            p0 = transl2(end)

            th = s * th0
            pr = s * p0
        else:
            #	TRINTERP2(T0, T1, s)
            assert start.shape == end.shape, 'both matrices must be same shape'

            th0 = math.atan2(start[1, 0], start[0, 0])
            th1 = math.atan2(end[1, 0], end[0, 0])

            p0 = transl2(start)
            p1 = transl2(end)

            pr = p0 * (1 - s) + s * p1
            th = th0 * (1 - s) + s * th1

        return trn.rt2tr(rot2(th), pr)
    else:
        return ValueError('Argument must be SO(2) or SE(2)')


def trprint2(T, label=None, file=sys.stdout, fmt='{:8.2g}', unit='deg'):
    """
    Compact display of SO(2) or SE(2) matrices

    :param T: matrix to format
    :type T: numpy.ndarray, shape=(2,2) or (3,3)
    :param label: text label to put at start of line
    :type label: str
    :param file: file to write formatted string to
    :type file: str
    :param fmt: conversion format for each number
    :type fmt: str
    :param unit: angular units: 'rad' [default], or 'deg'
    :type unit: str
    :return: formatted string
    :rtype: str

    The matrix is formatted and written to ``file`` and the
    string is returned.  To suppress writing to a file, set ``file=None``.

    - ``trprint2(R)`` displays the SO(2) rotation matrix in a compact
      single-line format and returns the string::

        [LABEL:] THETA UNIT

    - ``trprint2(T)`` displays the SE(2) homogoneous transform in a compact
      single-line format and returns the string::

        [LABEL:] [t=X, Y;] THETA UNIT

    Example::

        >>> T = transl2(1,2) @ trot2(0.3)
        >>> trprint2(a, file=None, label='T')
        'T: t =        1,        2;       17 deg'

    :seealso: trprint
    """

    s = ''

    if label is not None:
        s += '{:s}: '.format(label)

    # print the translational part if it exists
    if ishom2(T):
        s += 't = {};'.format(_vec2s(fmt, transl2(T)))

    angle = math.atan2(T[1, 0], T[0, 0])
    if unit == 'deg':
        angle *= 180.0 / math.pi
    s += ' {} {}'.format(_vec2s(fmt, [angle]), unit)

    if file:
        print(s, file=file)
    return s


def _vec2s(fmt, v):
    v = [x if np.abs(x) > 100 * _eps else 0.0 for x in v]
    return ', '.join([fmt.format(x) for x in v])


try:
    import matplotlib.pyplot as plt
    _matplotlib_exists = True

except ImportError:  # pragma: no cover
    def trplot2(*args, **kwargs):  # pylint: disable=unused-argument,missing-function-docstring
        print('matplotlib is not installed: pip install matplotlib')
    _matplotlib_exists = False

if _matplotlib_exists:

    def trplot2(T, axes=None, block=True, dims=None, color='blue', frame=None, # pylint: disable=unused-argument,function-redefined
                textcolor=None, labels=('X', 'Y'), length=1, arrow=True,
                rviz=False, wtl=0.2, width=1, d1=0.05, d2=1.15, **kwargs):  
        """
        Plot a 2D coordinate frame

        :param T: an SO(3) or SE(3) pose to be displayed as coordinate frame
        :type: numpy.ndarray, shape=(2,2) or (3,3)
        :param axes: the axes to plot into, defaults to current axes
        :type axes: Axes3D reference
        :param block: run the GUI main loop until all windows are closed, default True
        :type block: bool
        :param dims: dimension of plot volume as [xmin, xmax, ymin, ymax]
        :type dims: array_like
        :param color: color of the lines defining the frame
        :type color: str
        :param textcolor: color of text labels for the frame, default color of lines above
        :type textcolor: str
        :param frame: label the frame, name is shown below the frame and as subscripts on the frame axis labels
        :type frame: str
        :param labels: labels for the axes, defaults to X, Y and Z
        :type labels: 3-tuple of strings
        :param length: length of coordinate frame axes, default 1
        :type length: float
        :param arrow: show arrow heads, default True
        :type arrow: bool
        :param wtl: width-to-length ratio for arrows, default 0.2
        :type wtl: float
        :param rviz: show Rviz style arrows, default False
        :type rviz: bool
        :param projection: 3D projection: ortho [default] or persp
        :type projection: str
        :param width: width of lines, default 1
        :type width: float
        :param d1: distance of frame axis label text from origin, default 1.15
        :type d2: distance of frame label text from origin, default 0.05
        :return: axes containing the frame
        :rtype: AxesSubplot

        Adds a 2D coordinate frame represented by the SO(2) or SE(2) matrix to the current axes.

        - If no current figure, one is created
        - If current figure, but no axes, a 3d Axes is created

        Examples:

             trplot2(T, frame='A')
             trplot2(T, frame='A', color='green')
             trplot2(T1, 'labels', 'AB');

        """

        # TODO
        # animation
        # style='line', 'arrow', 'rviz'

        # check input types
        if isrot2(T, check=True):
            T = trn.r2t(T)
        else:
            assert ishom2(T, check=True)

        if axes is None:
            # create an axes
            fig = plt.gcf()
            if fig.axes == []:
                # no axes in the figure, create a 3D axes
                ax = plt.gca()

                if dims is None:
                    ax.autoscale(enable=True, axis='both')
                else:
                    if len(dims) == 2:
                        dims = dims * 2
                    ax.set_xlim(dims[0:2])
                    ax.set_ylim(dims[2:4])
                ax.set_aspect('equal')
                ax.set_xlabel(labels[0])
                ax.set_ylabel(labels[1])
            else:
                # reuse an existing axis
                ax = plt.gca()
        else:
            ax = axes

        # create unit vectors in homogeneous form
        o = T @ np.array([0, 0, 1])
        x = T @ np.array([1, 0, 1]) * length
        y = T @ np.array([0, 1, 1]) * length

        # draw the axes

        if rviz:
            ax.plot([o[0], x[0]], [o[1], x[1]], color='red', linewidth=5 * width)
            ax.plot([o[0], y[0]], [o[1], y[1]], color='lime', linewidth=5 * width)
        elif arrow:
            ax.quiver(o[0], o[1], x[0] - o[0], x[1] - o[1], angles='xy', scale_units='xy', scale=1, linewidth=width, facecolor=color, edgecolor=color)
            ax.quiver(o[0], o[1], y[0] - o[0], y[1] - o[1], angles='xy', scale_units='xy', scale=1, linewidth=width, facecolor=color, edgecolor=color)
            # plot an invisible point at the end of each arrow to allow auto-scaling to work
            ax.scatter(x=[o[0], x[0], y[0]], y=[o[1], x[1], y[1]], s=[20, 0, 0])
        else:
            ax.plot([o[0], x[0]], [o[1], x[1]], color=color, linewidth=width)
            ax.plot([o[0], y[0]], [o[1], y[1]], color=color, linewidth=width)

        # label the frame
        if frame:
            if textcolor is not None:
                color = textcolor

            o1 = T @ np.array([-d1, -d1, 1])
            ax.text(o1[0], o1[1], r'$\{' + frame + r'\}$', color=color, verticalalignment='top', horizontalalignment='center')

            # add the labels to each axis

            x = (x - o) * d2 + o
            y = (y - o) * d2 + o

            ax.text(x[0], x[1], "$%c_{%s}$" % (labels[0], frame), color=color, horizontalalignment='center', verticalalignment='center')
            ax.text(y[0], y[1], "$%c_{%s}$" % (labels[1], frame), color=color, horizontalalignment='center', verticalalignment='center')

        if block:
            # calling this at all, causes FuncAnimation to fail so when invoked from tranimate2 skip this bit
            plt.show(block=block)
        return ax

    def tranimate2(T, **kwargs):
        """
        Animate a 2D coordinate frame

        :param T: an SO(2) or SE(2) pose to be displayed as coordinate frame
        :type: numpy.ndarray, shape=(2,2) or (3,3)
        :param nframes: number of steps in the animation [defaault 100]
        :type nframes: int
        :param repeat: animate in endless loop [default False]
        :type repeat: bool
        :param interval: number of milliseconds between frames [default 50]
        :type interval: int
        :param movie: name of file to write MP4 movie into
        :type movie: str

        Animates a 2D coordinate frame moving from the world frame to a frame represented by the SO(2) or SE(2) matrix to the current axes.

        - If no current figure, one is created
        - If current figure, but no axes, a 3d Axes is created


        Examples:

             tranimate2(transl(1,2)@trot2(1), frame='A', arrow=False, dims=[0, 5])
             tranimate2(transl(1,2)@trot2(1), frame='A', arrow=False, dims=[0, 5], movie='spin.mp4')
        """
        anim = animate.Animate2(**kwargs)
        anim.trplot2(T, **kwargs)
        anim.run(**kwargs)


if __name__ == '__main__':  # pragma: no cover
    import pathlib

    # trplot2( transl2(1,2), frame='A', rviz=True, width=1)
    # trplot2( transl2(3,1), color='red', arrow=True, width=3, frame='B')
    # trplot2( transl2(4, 3)@trot2(math.pi/3), color='green', frame='c')
    # plt.grid(True)

    exec(open(pathlib.Path(__file__).parent.absolute() / "test_transforms.py").read())  # pylint: disable=exec-used
