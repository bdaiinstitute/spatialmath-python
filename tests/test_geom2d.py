#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 14:37:24 2020

@author: corkep
"""

from spatialmath.geom2d import *
from spatialmath.pose2d import SE2

import pytest
import sys
import numpy.testing as nt
import spatialmath.base as smb


class Polygon2Test:
    # Primitives
    def test_constructor1(self):
        p = Polygon2([(1, 2), (3, 2), (2, 4)])
        assert isinstance(p, Polygon2)
        assert len(p) == 3
        assert str(p) == "Polygon2 with 4 vertices"
        nt.assert_array_equal(p.vertices(), np.array([[1, 3, 2], [2, 2, 4]]))
        nt.assert_array_equal(
            p.vertices(unique=False), np.array([[1, 3, 2, 1], [2, 2, 4, 2]])
        )

    def test_methods(self):
        p = Polygon2(np.array([[-1, 1, 1, -1], [-1, -1, 1, 1]]))

        assert p.area() == 4
        assert p.moment(0, 0) == 4
        assert p.moment(1, 0) == 0
        assert p.moment(0, 1) == 0
        nt.assert_array_equal(p.centroid(), np.r_[0, 0])

        assert p.radius() == np.sqrt(2)
        nt.assert_array_equal(p.bbox(), np.r_[-1, -1, 1, 1])

    def test_contains(self):
        p = Polygon2(np.array([[-1, 1, 1, -1], [-1, -1, 1, 1]]))
        assert p.contains([0, 0], radius=1e-6)
        assert p.contains([1, 0], radius=1e-6)
        assert p.contains([-1, 0], radius=1e-6)
        assert p.contains([0, 1], radius=1e-6)
        assert p.contains([0, -1], radius=1e-6)

        assert not p.contains([0, 1.1], radius=1e-6)
        assert not p.contains([0, -1.1], radius=1e-6)
        assert not p.contains([1.1, 0], radius=1e-6)
        assert not p.contains([-1.1, 0], radius=1e-6)

        assert p.contains(np.r_[0, -1], radius=1e-6)
        assert not p.contains(np.r_[0, 1.1], radius=1e-6)

    def test_transform(self):
        p = Polygon2(np.array([[-1, 1, 1, -1], [-1, -1, 1, 1]]))

        p = p.transformed(SE2(2, 3))

        assert p.area() == 4
        assert p.moment(0 == 0, 4)
        assert p.moment(1 == 0, 8)
        assert p.moment(0 == 1, 12)
        nt.assert_array_equal(p.centroid(), np.r_[2, 3])

    def test_intersect(self):
        p1 = Polygon2(np.array([[-1, 1, 1, -1], [-1, -1, 1, 1]]))

        p2 = p1.transformed(SE2(2, 3))
        assert not p1.intersects(p2)

        p2 = p1.transformed(SE2(1, 1))
        assert p1.intersects(p2)

        assert p1.intersects(p1)

    def test_intersect_line(self):
        p = Polygon2(np.array([[-1, 1, 1, -1], [-1, -1, 1, 1]]))

        l = Line2.Join((-10, 0), (10, 0))
        assert p.intersects(l)

        l = Line2.Join((-10, 1.1), (10, 1.1))
        assert not p.intersects(l)

    @pytest.mark.skipif(
        sys.platform.startswith("darwin") and sys.version_info < (3, 11),
        reason="tkinter bug with mac",
    )
    def test_plot(self):
        p = Polygon2(np.array([[-1, 1, 1, -1], [-1, -1, 1, 1]]))
        p.plot()

        p.animate(SE2(1, 2))

    def test_edges(self):
        p = Polygon2([(1, 2), (3, 2), (2, 4)])
        e = p.edges()

        e = list(e)
        nt.assert_equal(e[0], ((1, 2), (3, 2)))
        nt.assert_equal(e[1], ((3, 2), (2, 4)))
        nt.assert_equal(e[2], ((2, 4), (1, 2)))

    # p.move(SE2(0, 0, 0.7))


class Line2Test:
    def test_constructor(self):
        l = Line2([1, 2, 3])
        assert str(l) == "Line2: [1. 2. 3.]"

        l = Line2.Join((0, 0), (1, 2))
        nt.assert_equal(l.line, [-2, 1, 0])

        l = Line2.General(2, 1)
        nt.assert_equal(l.line, [2, -1, 1])

    def test_contains(self):
        l = Line2.Join((0, 0), (1, 2))

        assert l.contains((0, 0))
        assert l.contains((1, 2))
        assert l.contains((2, 4))

    def test_intersect(self):
        l1 = Line2.Join((0, 0), (2, 0))  # y = 0
        l2 = Line2.Join((0, 1), (2, 1))  # y = 1
        assert not l1.intersect(l2)

        l2 = Line2.Join((2, 1), (2, -1))  # x = 2
        assert l1.intersect(l2)

    def test_intersect_segment(self):
        l1 = Line2.Join((0, 0), (2, 0))  # y = 0
        assert not l1.intersect_segment((2, 1), (2, 3))
        assert l1.intersect_segment((2, 1), (2, -1))


class EllipseTest:
    def test_constructor(self):
        E = np.array([[1, 1], [1, 3]])
        e = Ellipse(E=E)
        nt.assert_almost_equal(e.E, E)
        nt.assert_almost_equal(e.centre, [0, 0])
        assert e.theta == pytest.approx(1.1780972450961724)

        e = Ellipse(radii=(1, 2), theta=0)
        nt.assert_almost_equal(e.E, np.diag([1, 0.25]))
        nt.assert_almost_equal(e.centre, [0, 0])
        nt.assert_almost_equal(e.radii, [1, 2])
        assert e.theta == pytest.approx(0)

        e = Ellipse(radii=(1, 2), theta=np.pi / 2)
        nt.assert_almost_equal(e.E, np.diag([0.25, 1]))
        nt.assert_almost_equal(e.centre, [0, 0])
        nt.assert_almost_equal(e.radii, [2, 1])
        assert e.theta == pytest.approx(np.pi / 2)

        E = np.array([[1, 1], [1, 3]])
        e = Ellipse(E=E, centre=[3, 4])
        nt.assert_almost_equal(e.E, E)
        nt.assert_almost_equal(e.centre, [3, 4])
        assert e.theta == pytest.approx(1.1780972450961724)

        e = Ellipse(radii=(1, 2), theta=0, centre=[3, 4])
        nt.assert_almost_equal(e.E, np.diag([1, 0.25]))
        nt.assert_almost_equal(e.centre, [3, 4])
        nt.assert_almost_equal(e.radii, [1, 2])
        assert e.theta == pytest.approx(0)

    def test_Polynomial(self):
        e = Ellipse.Polynomial([2, 3, 1, 0, 0, -1])
        nt.assert_almost_equal(e.E, np.array([[2, 0.5], [0.5, 3]]))
        nt.assert_almost_equal(e.centre, [0, 0])

    def test_FromPerimeter(self):
        eref = Ellipse(radii=(1, 2), theta=0, centre=[0, 0])
        p = eref.points()

        e = Ellipse.FromPerimeter(p)
        nt.assert_almost_equal(e.radii, eref.radii)
        nt.assert_almost_equal(e.centre, eref.centre)
        nt.assert_almost_equal(e.theta, eref.theta)

        ##
        eref = Ellipse(radii=(1, 2), theta=0, centre=[3, 4])
        p = eref.points()

        e = Ellipse.FromPerimeter(p)
        nt.assert_almost_equal(e.radii, eref.radii)
        nt.assert_almost_equal(e.centre, eref.centre)
        nt.assert_almost_equal(e.theta, eref.theta)

        ##
        eref = Ellipse(radii=(1, 2), theta=np.pi / 4, centre=[3, 4])
        p = eref.points()

        e = Ellipse.FromPerimeter(p)
        nt.assert_almost_equal(e.radii, eref.radii)
        nt.assert_almost_equal(e.centre, eref.centre)
        nt.assert_almost_equal(e.theta, eref.theta)

    def test_FromPoints(self):
        eref = Ellipse(radii=(1, 2), theta=np.pi / 2, centre=(3, 4))
        rng = np.random.default_rng(0)

        # create 200 random points inside the ellipse
        x = []
        while len(x) < 200:
            p = rng.uniform(low=1, high=6, size=(2, 1))
            if eref.contains(p):
                x.append(p)
        x = np.hstack(x)  # create 2 x 50 array

        e = Ellipse.FromPoints(x)
        nt.assert_almost_equal(e.radii, eref.radii, decimal=1)
        nt.assert_almost_equal(e.centre, eref.centre, decimal=1)
        nt.assert_almost_equal(e.theta, eref.theta, decimal=1)

    def test_misc(self):
        e = Ellipse(radii=(1, 2), theta=np.pi / 2)
        assert isinstance(str(e), str)

        assert e.area == pytest.approx(np.pi * 2)

        e = Ellipse(radii=(1, 2), theta=0)
        assert e.contains((0, 0))
        assert e.contains((1, 0))
        assert e.contains((-1, 0))
        assert e.contains((0, 2))
        assert e.contains((0, -2))

        assert not e.contains((1.1, 0))
        assert not e.contains((-1.1, 0))
        assert not e.contains((0, 2.1))
        assert not e.contains((0, -2.1))

        assert e.contains(np.array([[0, 0], [3, 3]]).T) == [True, False]
