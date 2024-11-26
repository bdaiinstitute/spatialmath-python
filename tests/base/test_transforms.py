#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 14:19:04 2020

@author: corkep

"""


import numpy as np
import numpy.testing as nt
from math import pi
import math
from scipy.linalg import logm, expm

from spatialmath.base import *
from spatialmath.base import sym

import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt


class TestLie:
    def test_vex(self):
        S = np.array([[0, -3], [3, 0]])

        nt.assert_array_almost_equal(vex(S), np.array([3]))
        nt.assert_array_almost_equal(vex(-S), np.array([-3]))

        S = np.array([[0, -3, 2], [3, 0, -1], [-2, 1, 0]])

        nt.assert_array_almost_equal(vex(S), np.array([1, 2, 3]))
        nt.assert_array_almost_equal(vex(-S), -np.array([1, 2, 3]))

    def test_skew(self):
        R = skew(3)
        nt.assert_equal(isrot2(R, check=False), True)  # check size
        nt.assert_array_almost_equal(np.linalg.norm(R.T + R), 0)  # check is skew
        nt.assert_array_almost_equal(
            vex(R), np.array([3])
        )  # check contents, vex already verified

        R = skew([1, 2, 3])
        nt.assert_equal(isrot(R, check=False), True)  # check size
        nt.assert_array_almost_equal(np.linalg.norm(R.T + R), 0)  # check is skew
        nt.assert_array_almost_equal(
            vex(R), np.array([1, 2, 3])
        )  # check contents, vex already verified

    def test_vexa(self):

        S = np.array([[0, -3, 1], [3, 0, 2], [0, 0, 0]])
        nt.assert_array_almost_equal(vexa(S), np.array([1, 2, 3]))

        S = np.array([[0, 3, -1], [-3, 0, 2], [0, 0, 0]])
        nt.assert_array_almost_equal(vexa(S), np.array([-1, 2, -3]))

        S = np.array([[0, -6, 5, 1], [6, 0, -4, 2], [-5, 4, 0, 3], [0, 0, 0, 0]])
        nt.assert_array_almost_equal(vexa(S), np.array([1, 2, 3, 4, 5, 6]))

        S = np.array([[0, 6, 5, 1], [-6, 0, 4, -2], [-5, -4, 0, 3], [0, 0, 0, 0]])
        nt.assert_array_almost_equal(vexa(S), np.array([1, -2, 3, -4, 5, -6]))

    def test_skewa(self):
        T = skewa([3, 4, 5])
        nt.assert_equal(ishom2(T, check=False), True)  # check size
        R = t2r(T)
        nt.assert_equal(np.linalg.norm(R.T + R), 0)  # check is skew
        nt.assert_array_almost_equal(
            vexa(T), np.array([3, 4, 5])
        )  # check contents, vexa already verified

        T = skewa([1, 2, 3, 4, 5, 6])
        nt.assert_equal(ishom(T, check=False), True)  # check size
        R = t2r(T)
        nt.assert_equal(np.linalg.norm(R.T + R), 0)  # check is skew
        nt.assert_array_almost_equal(
            vexa(T), np.array([1, 2, 3, 4, 5, 6])
        )  # check contents, vexa already verified

    def test_trlog(self):

        # %%% SO(3) tests
        # zero rotation case
        nt.assert_array_almost_equal(trlog(np.eye(3)), skew([0, 0, 0]))
        nt.assert_array_almost_equal(trlog(np.eye(3), twist=True), np.r_[0, 0, 0])

        # rotation by pi case
        nt.assert_array_almost_equal(trlog(rotx(pi)), skew([pi, 0, 0]))
        nt.assert_array_almost_equal(trlog(roty(pi)), skew([0, pi, 0]))
        nt.assert_array_almost_equal(trlog(rotz(pi)), skew([0, 0, pi]))

        nt.assert_array_almost_equal(trlog(rotx(pi), twist=True), np.r_[pi, 0, 0])
        nt.assert_array_almost_equal(trlog(roty(pi), twist=True), np.r_[0, pi, 0])
        nt.assert_array_almost_equal(trlog(rotz(pi), twist=True), np.r_[0, 0, pi])

        # general case
        nt.assert_array_almost_equal(trlog(rotx(0.2)), skew([0.2, 0, 0]))
        nt.assert_array_almost_equal(trlog(roty(0.3)), skew([0, 0.3, 0]))
        nt.assert_array_almost_equal(trlog(rotz(0.4)), skew([0, 0, 0.4]))

        nt.assert_array_almost_equal(trlog(rotx(0.2), twist=True), np.r_[0.2, 0, 0])
        nt.assert_array_almost_equal(trlog(roty(0.3), twist=True), np.r_[0, 0.3, 0])
        nt.assert_array_almost_equal(trlog(rotz(0.4), twist=True), np.r_[0, 0, 0.4])

        R = rotx(0.2) @ roty(0.3) @ rotz(0.4)
        nt.assert_array_almost_equal(trlog(R), logm(R))
        nt.assert_array_almost_equal(trlog(R, twist=True), vex(logm(R)))

        # SE(3) tests

        # pure translation
        nt.assert_array_almost_equal(
            trlog(transl([1, 2, 3])),
            np.array([[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 3], [0, 0, 0, 0]]),
        )
        nt.assert_array_almost_equal(
            trlog(transl([1, 2, 3]), twist=True), np.r_[1, 2, 3, 0, 0, 0]
        )

        # pure rotation
        # rotation by pi case
        nt.assert_array_almost_equal(trlog(trotx(pi)), skewa([0, 0, 0, pi, 0, 0]))
        nt.assert_array_almost_equal(trlog(troty(pi)), skewa([0, 0, 0, 0, pi, 0]))
        nt.assert_array_almost_equal(trlog(trotz(pi)), skewa([0, 0, 0, 0, 0, pi]))

        nt.assert_array_almost_equal(
            trlog(trotx(pi), twist=True), np.r_[0, 0, 0, pi, 0, 0]
        )
        nt.assert_array_almost_equal(
            trlog(troty(pi), twist=True), np.r_[0, 0, 0, 0, pi, 0]
        )
        nt.assert_array_almost_equal(
            trlog(trotz(pi), twist=True), np.r_[0, 0, 0, 0, 0, pi]
        )

        # general case
        nt.assert_array_almost_equal(trlog(trotx(0.2)), skewa([0, 0, 0, 0.2, 0, 0]))
        nt.assert_array_almost_equal(trlog(troty(0.3)), skewa([0, 0, 0, 0, 0.3, 0]))
        nt.assert_array_almost_equal(trlog(trotz(0.4)), skewa([0, 0, 0, 0, 0, 0.4]))

        nt.assert_array_almost_equal(
            trlog(trotx(0.2), twist=True), np.r_[0, 0, 0, 0.2, 0, 0]
        )
        nt.assert_array_almost_equal(
            trlog(troty(0.3), twist=True), np.r_[0, 0, 0, 0, 0.3, 0]
        )
        nt.assert_array_almost_equal(
            trlog(trotz(0.4), twist=True), np.r_[0, 0, 0, 0, 0, 0.4]
        )

        # mixture
        T = transl([1, 2, 3]) @ trotx(0.3)
        nt.assert_array_almost_equal(trlog(T), logm(T))
        nt.assert_array_almost_equal(trlog(T, twist=True), vexa(logm(T)))

        T = transl([1, 2, 3]) @ troty(0.3)
        nt.assert_array_almost_equal(trlog(T), logm(T))
        nt.assert_array_almost_equal(trlog(T, twist=True), vexa(logm(T)))

    # def test_trlog2(self):

    #     #%%% SO(2) tests
    #     # zero rotation case
    #     nt.assert_array_almost_equal(trlog2( np.eye(2) ), skew([0]))

    #     # rotation by pi case
    #     nt.assert_array_almost_equal(trlog2( rot2(pi) ), skew([pi]))

    #     # general case
    #     nt.assert_array_almost_equal(trlog2( rotx(0.2) ), skew([0.2]))

    #     #%% SE(3) tests

    #     # pure translation
    #     nt.assert_array_almost_equal(trlog2( transl2([1, 2]) ), np.array([[0, 0, 1], [ 0, 0, 2], [ 0, 0, 0]]))

    #     # pure rotation
    #     # rotation by pi case
    #     nt.assert_array_almost_equal(trlog( trot2(pi) ), skewa([0, 0, pi]))

    #     # general case
    #     nt.assert_array_almost_equal(trlog( trot2(0.2) ), skewa([0, 0, 0.2]))

    #     # mixture
    #     T = transl([1, 2, 3]) @ trot2(0.3)
    #     nt.assert_array_almost_equal(trlog2(T), logm(T))
    # TODO

    def test_trexp(self):

        # %% SO(3) tests

        # % so(3)

        # zero rotation case
        nt.assert_array_almost_equal(trexp(skew([0, 0, 0])), np.eye(3))
        nt.assert_array_almost_equal(trexp([0, 0, 0]), np.eye(3))

        # % so(3), theta

        # rotation by pi case
        nt.assert_array_almost_equal(trexp(skew([pi, 0, 0])), rotx(pi))
        nt.assert_array_almost_equal(trexp(skew([0, pi, 0])), roty(pi))
        nt.assert_array_almost_equal(trexp(skew([0, 0, pi])), rotz(pi))

        # general case
        nt.assert_array_almost_equal(trexp(skew([0.2, 0, 0])), rotx(0.2))
        nt.assert_array_almost_equal(trexp(skew([0, 0.3, 0])), roty(0.3))
        nt.assert_array_almost_equal(trexp(skew([0, 0, 0.4])), rotz(0.4))

        nt.assert_array_almost_equal(trexp(skew([1, 0, 0]), 0.2), rotx(0.2))
        nt.assert_array_almost_equal(trexp(skew([0, 1, 0]), 0.3), roty(0.3))
        nt.assert_array_almost_equal(trexp(skew([0, 0, 1]), 0.4), rotz(0.4))

        nt.assert_array_almost_equal(trexp([1, 0, 0], 0.2), rotx(0.2))
        nt.assert_array_almost_equal(trexp([0, 1, 0], 0.3), roty(0.3))
        nt.assert_array_almost_equal(trexp([0, 0, 1], 0.4), rotz(0.4))

        nt.assert_array_almost_equal(trexp(np.r_[1, 0, 0] * 0.2), rotx(0.2))
        nt.assert_array_almost_equal(trexp(np.r_[0, 1, 0] * 0.3), roty(0.3))
        nt.assert_array_almost_equal(trexp(np.r_[0, 0, 1] * 0.4), rotz(0.4))

        # %% SE(3) tests

        # zero motion case
        nt.assert_array_almost_equal(trexp(skewa([0, 0, 0, 0, 0, 0])), np.eye(4))
        nt.assert_array_almost_equal(trexp([0, 0, 0, 0, 0, 0]), np.eye(4))

        # % sigma = se(3)
        # pure translation
        nt.assert_array_almost_equal(
            trexp(skewa([1, 2, 3, 0, 0, 0])), transl([1, 2, 3])
        )
        nt.assert_array_almost_equal(trexp(skewa([0, 0, 0, 0.2, 0, 0])), trotx(0.2))
        nt.assert_array_almost_equal(trexp(skewa([0, 0, 0, 0, 0.3, 0])), troty(0.3))
        nt.assert_array_almost_equal(trexp(skewa([0, 0, 0, 0, 0, 0.4])), trotz(0.4))

        nt.assert_array_almost_equal(trexp([1, 2, 3, 0, 0, 0]), transl([1, 2, 3]))
        nt.assert_array_almost_equal(trexp([0, 0, 0, 0.2, 0, 0]), trotx(0.2))
        nt.assert_array_almost_equal(trexp([0, 0, 0, 0, 0.3, 0]), troty(0.3))
        nt.assert_array_almost_equal(trexp([0, 0, 0, 0, 0, 0.4]), trotz(0.4))

        # mixture
        S = skewa([1, 2, 3, 0.1, -0.2, 0.3])
        nt.assert_array_almost_equal(trexp(S), expm(S))

        # twist vector
        # nt.assert_array_almost_equal(trexp( double(Twist(T))), T)

        # (sigma, theta)
        nt.assert_array_almost_equal(
            trexp(skewa([1, 0, 0, 0, 0, 0]), 2), transl([2, 0, 0])
        )
        nt.assert_array_almost_equal(
            trexp(skewa([0, 1, 0, 0, 0, 0]), 2), transl([0, 2, 0])
        )
        nt.assert_array_almost_equal(
            trexp(skewa([0, 0, 1, 0, 0, 0]), 2), transl([0, 0, 2])
        )

        nt.assert_array_almost_equal(trexp(skewa([0, 0, 0, 1, 0, 0]), 0.2), trotx(0.2))
        nt.assert_array_almost_equal(trexp(skewa([0, 0, 0, 0, 1, 0]), 0.2), troty(0.2))
        nt.assert_array_almost_equal(trexp(skewa([0, 0, 0, 0, 0, 1]), 0.2), trotz(0.2))

        # (twist, theta)
        # nt.assert_array_almost_equal(trexp(Twist('R', [1, 0, 0], [0, 0, 0]).S, 0.3), trotx(0.3))

        T = transl([1, 2, 3]) @ trotz(0.3)
        nt.assert_array_almost_equal(trexp(trlog(T)), T)

    def test_trexp2(self):

        # % so(2)

        # zero rotation case
        nt.assert_array_almost_equal(trexp2(skew([0])), np.eye(2))
        nt.assert_array_almost_equal(trexp2(skew(0)), np.eye(2))

        # % so(2), theta

        # rotation by pi case
        nt.assert_array_almost_equal(trexp2(skew(pi)), rot2(pi))

        # general case
        nt.assert_array_almost_equal(trexp2(skew(0.2)), rot2(0.2))

        nt.assert_array_almost_equal(trexp2(1, 0.2), rot2(0.2))

        # %% SE(3) tests

        # % sigma = se(3)
        # pure translation
        nt.assert_array_almost_equal(trexp2(skewa([1, 2, 0])), transl2([1, 2]))

        nt.assert_array_almost_equal(trexp2([0, 0, 0.2]), trot2(0.2))

        # mixture
        S = skewa([1, 2, 0.3])
        nt.assert_array_almost_equal(trexp2(S), expm(S))

        # twist vector
        # nt.assert_array_almost_equal(trexp( double(Twist(T))), T)

        # (sigma, theta)
        nt.assert_array_almost_equal(trexp2(skewa([1, 0, 0]), 2), transl2([2, 0]))
        nt.assert_array_almost_equal(trexp2(skewa([0, 1, 0]), 2), transl2([0, 2]))

        nt.assert_array_almost_equal(trexp2(skewa([0, 0, 1]), 0.2), trot2(0.2))

        # (twist, theta)
        # nt.assert_array_almost_equal(trexp(Twist('R', [1, 0, 0], [0, 0, 0]).S, 0.3), trotx(0.3))

        # T = transl2([1, 2])@trot2(0.3)
        # nt.assert_array_almost_equal(trexp2(trlog2(T)), T)
        # TODO

    def test_trnorm(self):
        T0 = transl(-1, -2, -3) @ trotx(-0.3)
        nt.assert_array_almost_equal(trnorm(T0), T0)
