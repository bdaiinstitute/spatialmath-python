# Part of Spatial Math Toolbox for Python
# Copyright (c) 2000 Peter Corke
# MIT Licence, see details in top-level file: LICENCE

# matplotlib inline

# line.set_data()
# text.set_position()
# quiver.set_offsets(), quiver.set_UVC()
# FancyArrow.set_xy()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from spatialmath import base


class Animate:
    """
    Animate objects for matplotlib 3d

    An instance of this class behaves like an Axes3D and supports proxies for

    - ``plot``
    - ``quiver``
    - ``text``

    which renders them and also places corresponding objects into a display
    list. These objects are ``Line``, ``Quiver`` and ``Text``.  Only these
    primitives will be animated.

    The objects are all drawn relative to the origin, and will be transformed
    according to the transform that is being animated.

    Example::

        anim = animate.Animate(dims=[0,2]) # set up the 3D axes
        anim.trplot(T, frame='A', color='green')  # draw the frame
        anim.run(loop=True)  # animate it
    """

    def __init__(self, axes=None, dims=None, projection='ortho', labels=('X', 'Y', 'Z'), **kwargs):
        """
        Construct an Animate object

        :param axes: the axes to plot into, defaults to current axes
        :type axes: Axes3D reference
        :param dims: dimension of plot volume as [xmin, xmax, ymin, ymax,
            zmin, zmax]. If dims is [min, max] those limits are applied
            to the x-, y- and z-axes.
        :type dims: array_like
        :param projection: 3D projection: ortho [default] or persp
        :type projection: str
        :param labels: labels for the axes, defaults to X, Y and Z
        :type labels: 3-tuple of strings

        Will setup to plot into an existing or a new Axes3D instance.

        """
        self.displaylist = []

        if axes is None:
            # create an axes
            fig = plt.gcf()
            for a in fig.axes:
                if a.name != "3d":
                    fig.delete(a)
            if len(fig.axes) == 0:
                # no axes in the figure, create a 3D axes
                axes = fig.add_subplot(111, projection='3d', proj_type=projection)
                axes.set_xlabel(labels[0])
                axes.set_ylabel(labels[1])
                axes.set_zlabel(labels[2])
                axes.autoscale(enable=True, axis='both')
            else:
                # reuse an existing axis
                axes = plt.gca()

        if dims is not None:
            if len(dims) == 2:
                dims = dims * 3
            axes.set_xlim(dims[0:2])
            axes.set_ylim(dims[2:4])
            axes.set_zlim(dims[4:6])
            # ax.set_aspect('equal')

        self.ax = axes

        # TODO set flag for 2d or 3d axes, flag errors on the methods called later

    def trplot(self, end, start=None, **kwargs):
        """
        Define the transform to animate

        :param end: the final pose SO(3) or SE(3) to display as a coordinate frame
        :type end: numpy.ndarray, shape=(3,3) or (4,4)
        :param start: the initial pose SO(3) or SE(3) to display as a coordinate frame, defaults to null
        :type start: numpy.ndarray, shape=(3,3) or (4,4)
        :param start: an 

        Is polymorphic with ``base.trplot`` and accepts the same parameters.
        This sets up the animation but doesn't execute it.

        :seealso: :func:`run`

        """
        # stash the final value
        if base.isrot(end):
            self.end = base.r2t(end)
        else:
            self.end = end

        if start is None:
            self.start = np.identity(4)
        else:
            if base.isrot(start):
                self.start = base.r2t(start)
            else:
                self.start = start

        # draw axes at the origin
        base.trplot(self.start, axes=self, block=None, **kwargs)

    def run(self, movie=None, axes=None, repeat=False, interval=50, nframes=100, pause=0, **kwargs):
        """
        Run the animation

        :param axes: the axes to plot into, defaults to current axes
        :type axes: Axes3D reference
        :param repeat: animate in endless loop [default False]
        :type repeat: bool
        :param nframes: number of steps in the animation [defaault 100]
        :type nframes: int
        :param interval: number of milliseconds between frames [default 50]
        :type interval: int
        :param movie: name of file to write MP4 movie into
        :type movie: str

        Animates a 3D coordinate frame moving from the world frame to a frame
        represented by the SO(3) or SE(3) matrix to the current axes.

        Notes:

        - the ``movie`` option requires the ffmpeg package to be installed:
            ``conda install -c conda-forge ffmpeg``
        - invokes the draw() method of every object in the display list
        """

        def update(frame, a):
            T = base.trinterp(start=self.start, end=self.end, s=frame / nframes)
            a._draw(T)
            if frame == nframes - 1:
                a.done = True
            return a.artists()

        # blit leaves a trail and first frame
        if movie is not None:
            repeat = False

        self.done = False
        ani = animation.FuncAnimation(fig=plt.gcf(), func=update, frames=range(0, nframes), fargs=(self,), blit=False, interval=interval, repeat=repeat)
        if movie is None:
            while repeat or not self.done:
                plt.pause(0.1)
        else:
            # Set up formatting for the movie files
            print('creating movie', movie)
            FFwriter = animation.FFMpegWriter(fps=10, extra_args=['-vcodec', 'libx264'])
            ani.save(movie, writer=FFwriter)

    def __repr__(self):
        """
        Human readable version of the display list

        :param self: the animation
        :type self: Animate
        :returns: readable version of the display list
        :rtype: str
        """
        return ', '.join([x.type for x in self.displaylist])

    def artists(self):
        """
        List of artists that need to be updated

        :param self: the animation
        :type self: Animate
        :returns: list of artists
        :rtype: list
        """
        return [x.h for x in self.displaylist]

    def _draw(self, T):
        for x in self.displaylist:
            x.draw(T)

    # ------------------- plot()

    class _Line:

        def __init__(self, anim, h, xs, ys, zs):
            # form 4x2 matrix, columns are first/last point in homogeneous form
            self.p = np.vstack([xs, ys, zs, [1, 1]])
            self.h = h
            self.type = 'line'
            self.anim = anim

        def draw(self, T):
            p = T @ self.p
            self.h.set_data(p[0, :], p[1, :])
            self.h.set_3d_properties(p[2, :])

    def plot(self, x, y, z, *args, **kwargs):
        """
        Plot a polyline

        :param x: list of x-coordinates
        :type x: array_like
        :param y: list of y-coordinates
        :type y: array_like
        :param z: list of z-coordinates
        :type z: array_like

        Other arguments as accepted by the matplotlib method.

        All arrays must have the same length.

        :seealso: :func:`matplotlib.pyplot.plot`
        """

        h, = self.ax.plot(x, y, z, *args, **kwargs)
        self.displaylist.append(Animate._Line(self, h, x, y, z))

    # ------------------- quiver()

    class _Quiver:

        def __init__(self, anim, h):
            self.type = 'quiver'
            self.anim = anim
            # for matplotlib 3.1.x
            # ._segments3d is 3x2x3
            #   first index: line segment in the collection
            #   second index: 0 = start, 1 = end
            #   third index: x, y, z components
            # https://stackoverflow.com/questions/48911643/set-uvc-equivilent-for-a-3d-quiver-plot-in-matplotlib
            #
            # for matplotlib 3.3.x
            # ._segments3d is a 3-element list, each element is 2x3

            # turn to homogeneous form, with columns per point, alternating start, end

            if isinstance(h._segments3d, np.ndarray):
                self.p = np.vstack([h._segments3d.reshape(6, 3).T, np.ones((1, 6))])  # result is 4x6
            else:
                self.p = np.vstack([np.hstack([x.T for x in h._segments3d]), np.ones((1, 6))])
            self.h = h
            self.type = 'arrow'
            self.anim = anim

        def draw(self, T):
            p = T @ self.p

            # reshape it
            p = p[0:3, :].T.reshape(3, 2, 3)
            self.h.set_segments(p)

    def quiver(self, x, y, z, u, v, w, *args, **kwargs):
        """
        Plot a quiver

        :param x: list of base x-coordinates
        :type x: array_like
        :param y: list of base y-coordinates
        :type y: array_like
        :param z: list of base z-coordinates
        :type z: array_like
        :param u: list of vector x-coordinates
        :type u: array_like
        :param v: list of vector y-coordinates
        :type v: array_like
        :param w: list of vector z-coordinates
        :type w: array_like

        Draws a series of arrows, the bases defined by corresponding elements
        of (x,y,z) and the vector has components defined by corresponding
        elements of (u,v,w).

        Other arguments as accepted by the matplotlib method.

        :seealso: :func:`matplotlib.pyplot.quiver`
        """
        h = self.ax.quiver(x, y, z, u, v, w, *args, **kwargs)
        self.displaylist.append(Animate._Quiver(self, h))

    # ------------------- text()

    class _Text:

        def __init__(self, anim, h, x, y, z):
            self.type = 'text'
            self.h = h
            self.p = np.r_[x, y, z, 1]
            self.anim = anim

        def draw(self, T):
            p = T @ self.p
            # x2, y2, _ = proj3d.proj_transform(
            #   p[0], p[1], p[2], self.anim.ax.get_proj())
            # self.h.set_position((x2, y2))
            self.h.set_position((p[0], p[1]))
            self.h.set_3d_properties(z=p[2], zdir='x')

    def text(self, x, y, z, *args, **kwargs):
        """
        Plot text

        :param x: x-coordinate
        :type x: float
        :param y: float
        :type y: array_like
        :param z: z-coordinate
        :type z: float

        Other arguments as accepted by the matplotlib method.

        :seealso: :func:`matplotlib.pyplot.text`
        """
        h = self.ax.text3D(x, y, z, *args, **kwargs)
        self.displaylist.append(Animate._Text(self, h, x, y, z))

    # ------------------- scatter()

    def scatter(self, **kwargs):
        pass

    # ------------------- wrappers for Axes primitives

    def set_xlim(self, *args, **kwargs):
        self.ax.set_xlim(*args, **kwargs)

    def set_ylim(self, *args, **kwargs):
        self.ax.set_ylim(*args, **kwargs)

    def set_zlim(self, *args, **kwargs):
        self.ax.set_zlim(*args, **kwargs)

    def set_xlabel(self, *args, **kwargs):
        self.ax.set_xlabel(*args, **kwargs)

    def set_ylabel(self, *args, **kwargs):
        self.ax.set_ylabel(*args, **kwargs)

    def set_zlabel(self, *args, **kwargs):
        self.ax.set_zlabel(*args, **kwargs)


class Animate2:
    """
    Animate objects for matplotlib 2d

    An instance of this class behaves like an Axes3D and supports proxies for

    - ``plot``
    - ``quiver``
    - ``text``

    which renders them and also places corresponding objects into a display
    list. These objects are ``Line``, ``Quiver`` and ``Text``.  Only these
    primitives will be animated.

    The objects are all drawn relative to the origin, and will be transformed
    according to the transform that is being animated.

    Example::

        anim = animate.Animate(dims=[0,2]) # set up the 3D axes
        anim.trplot(T, frame='A', color='green')  # draw the frame
        anim.run(loop=True)  # animate it
    """

    def __init__(self, axes=None, dims=None, labels=('X', 'Y'), **kwargs):
        """
        Construct an Animate object

        :param axes: the axes to plot into, defaults to current axes
        :type axes: Axes3D reference
        :param dims: dimension of plot volume as [xmin, xmax, ymin, ymax,
            zmin, zmax]. If dims is [min, max] those limits are applied
            to the x-, y- and z-axes.
        :type dims: array_like
        :param projection: 3D projection: ortho [default] or persp
        :type projection: str
        :param labels: labels for the axes, defaults to X, Y and Z
        :type labels: 3-tuple of strings

        Will setup to plot into an existing or a new Axes3D instance.

        """
        self.displaylist = []

        if axes is None:
            # create an axes
            fig = plt.gcf()
            if fig.axes is None:
                # no axes in the figure, create a 3D axes
                axes = fig.add_subplot(111)
                axes.set_xlabel(labels[0])
                axes.set_ylabel(labels[1])
                axes.autoscale(enable=True, axis='both')
            else:
                # reuse an existing axis
                axes = plt.gca()

        if dims is not None:
            if len(dims) == 2:
                dims = dims * 2
            axes.set_xlim(dims[0:2])
            axes.set_ylim(dims[2:4])
            # ax.set_aspect('equal')

        self.ax = axes

        # set flag for 2d or 3d axes, flag errors on the methods called later

    def trplot2(self, end, start=None, **kwargs):
        """
        Define the transform to animate

        :param end: the final pose SO(2) or SE(2) to display as a coordinate frame
        :type end: numpy.ndarray, shape=(2,2) or (3,3)
        :param start: the initial pose SO(2) or SE(2) to display as a coordinate frame, defaults to null
        :type start: numpy.ndarray, shape=(2,2) or (3,3)

        Is polymorphic with ``base.trplot`` and accepts the same parameters.
        This sets up the animation but doesn't execute it.

        :seealso: :func:`run`

        """
        # stash the final value
        if base.isrot2(end):
            self.end = base.r2t(end)
        else:
            self.end = end

        if start is None:
            self.start = np.identity(3)
        else:
            if base.isrot2(start):
                self.start = base.r2t(start)
            else:
                self.start = start

        # draw axes at the origin
        base.trplot2(self.start, axes=self, block=False, **kwargs)

    def run(self, movie=None, axes=None, repeat=False, interval=50, nframes=100, **kwargs):
        """
        Run the animation

        :param axes: the axes to plot into, defaults to current axes
        :type axes: Axes3D reference
        :param nframes: number of steps in the animation [defaault 100]
        :type nframes: int
        :param repeat: animate in endless loop [default False]
        :type repeat: bool
        :param interval: number of milliseconds between frames [default 50]
        :type interval: int
        :param movie: name of file to write MP4 movie into
        :type movie: str

        Animates a 3D coordinate frame moving from the world frame to a frame
        represented by the SO(3) or SE(3) matrix to the current axes.

        Notes:

        - the ``movie`` option requires the ffmpeg package to be installed:
            ``conda install -c conda-forge ffmpeg``
        - invokes the draw() method of every object in the display list
        """

        def update(frame, a):
            T = base.trinterp2(start=self.start, end=self.end, s=frame / nframes)
            a._draw(T)
            if frame == nframes - 1:
                a.done = True
            return a.artists()

        # blit leaves a trail and first frame
        if movie is not None:
            repeat = False

        self.done = False
        ani = animation.FuncAnimation(fig=plt.gcf(), func=update, frames=range(0, nframes), fargs=(self,), blit=False, interval=interval, repeat=repeat)

        if movie is None:
            while repeat or not self.done:
                plt.pause(1)
        else:
            # Set up formatting for the movie files
            print('creating movie', movie)
            FFwriter = animation.FFMpegWriter(fps=10, extra_args=['-vcodec', 'libx264'])
            ani.save(movie, writer=FFwriter)

    def __repr__(self):
        """
        Human readable version of the display list

        :param self: the animation
        :type self: Animate
        :returns: readable version of the display list
        :rtype: str
        """
        return ', '.join([x.type for x in self.displaylist])

    def artists(self):
        """
        List of artists that need to be updated

        :param self: the animation
        :type self: Animate
        :returns: list of artists
        :rtype: list
        """
        return [x.h for x in self.displaylist]

    def _draw(self, T):
        for x in self.displaylist:
            x.draw(T)

    # ------------------- plot()

    class _Line:

        def __init__(self, anim, h, xs, ys):
            # form 3x2 matrix, columns are first/last point in homogeneous form
            self.p = np.vstack([xs, ys, [1, 1]])
            self.h = h
            self.type = 'line'
            self.anim = anim

        def draw(self, T):
            p = T @ self.p
            self.h.set_data(p[0, :], p[1, :])

    def plot(self, x, y, *args, **kwargs):
        """
        Plot a polyline

        :param x: list of x-coordinates
        :type x: array_like
        :param y: list of y-coordinates
        :type y: array_like

        Other arguments as accepted by the matplotlib method.

        All arrays must have the same length.

        :seealso: :func:`matplotlib.pyplot.plot`
        """

        h, = self.ax.plot(x, y, *args, **kwargs)
        self.displaylist.append(Animate2._Line(self, h, x, y))

    # ------------------- quiver()

    class _Quiver:

        def __init__(self, anim, h, x, y, u, v):
            self.type = 'quiver'
            self.anim = anim

            self.h = h
            self.type = 'arrow'
            self.anim = anim

            self.p = np.c_[u - x, v - y].T

        def draw(self, T):
            R, t = base.tr2rt(T)
            p = R @ self.p
            # specific to a single Quiver
            self.h.set_offsets(t)  # shift the origin
            self.h.set_UVC(p[0], p[1])

    def quiver(self, x, y, u, v, *args, **kwargs):
        """
        Plot a quiver

        :param x: list of base x-coordinates
        :type x: array_like
        :param y: list of base y-coordinates
        :type y: array_like
        :param u: list of vector x-coordinates
        :type u: array_like
        :param v: list of vector y-coordinates
        :type v: array_like


        Draws a series of arrows, the bases defined by corresponding elements
        of (x,y,z) and the vector has components defined by corresponding
        elements of (u,v,w).

        Other arguments as accepted by the matplotlib method.

        :seealso: :func:`matplotlib.pyplot.quiver`
        """
        h = self.ax.quiver(x, y, u, v, *args, **kwargs)
        self.displaylist.append(Animate2._Quiver(self, h, x, y, u, v))

    # ------------------- text()

    class _Text:

        def __init__(self, anim, h, x, y):
            self.type = 'text'
            self.h = h
            self.p = np.r_[x, y, 1]
            self.anim = anim

        def draw(self, T):
            p = T @ self.p
            # x2, y2, _ = proj3d.proj_transform(
            #   p[0], p[1], p[2], self.anim.ax.get_proj())
            # self.h.set_position((x2, y2))
            self.h.set_position((p[0], p[1]))

    def text(self, x, y, *args, **kwargs):
        """
        Plot text

        :param x: x-coordinate
        :type x: float
        :param y: float
        :type y: array_like
        :param z: z-coordinate
        :type z: float

        Other arguments as accepted by the matplotlib method.

        :seealso: :func:`matplotlib.pyplot.text`
        """
        h = self.ax.text(x, y, *args, **kwargs)
        self.displaylist.append(Animate2._Text(self, h, x, y))

    # ------------------- scatter()

    def scatter(self, **kwargs):
        pass

    # ------------------- wrappers for Axes primitives

    def set_xlim(self, *args, **kwargs):
        self.ax.set_xlim(*args, **kwargs)

    def set_ylim(self, *args, **kwargs):
        self.ax.set_ylim(*args, **kwargs)

    def set_xlabel(self, *args, **kwargs):
        self.ax.set_xlabel(*args, **kwargs)

    def set_ylabel(self, *args, **kwargs):
        self.ax.set_ylabel(*args, **kwargs)
