#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 14:19:04 2020

@author: corkep

"""

import numpy as np
import numpy.testing as nt
import pytest
import unittest
from math import pi
import math
from scipy.linalg import logm, expm

from spatialmath.base.transformsNd import *
from spatialmath.base.transforms3d import trotx, transl, rotx, isrot, ishom
from spatialmath.base.transforms2d import trot2, transl2, rot2, isrot2, ishom2

try:
    import sympy as sp

    _symbolics = True
    from spatialmath.base.symbolic import symbol
except ImportError:
    _symbolics = False
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt


class TestND(unittest.TestCase):
    def test_iseye(self):
        assert iseye(np.eye(1))
        assert iseye(np.eye(2))
        assert iseye(np.eye(3))
        assert iseye(np.eye(5))

        assert not iseye(2 * np.eye(3))
        assert not iseye(-np.eye(3))
        assert not iseye(np.array([[1, 0, 0], [0, 1, 0]]))
        assert not iseye(np.array([1, 0, 0]))

    def test_r2t(self):
        # 3D
        R = rotx(0.3)
        T = r2t(R)
        nt.assert_array_almost_equal(T[0:3, 3], np.r_[0, 0, 0])
        nt.assert_array_almost_equal(T[:3, :3], R)

        # 2D
        R = rot2(0.3)
        T = r2t(R)
        nt.assert_array_almost_equal(T[0:2, 2], np.r_[0, 0])
        nt.assert_array_almost_equal(T[:2, :2], R)

        with pytest.raises(ValueError):
            r2t(3)

        with pytest.raises(ValueError):
            r2t(np.eye(3, 4))
        
        _ = r2t(np.ones((3, 3)), check=False)
        with pytest.raises(ValueError):
            r2t(np.ones((3, 3)), check=True)

    @pytest.mark.skipif(not _symbolics, reason="sympy required")
    def test_r2t_sym(self):
        theta = symbol("theta")
        R = rot2(theta)
        T = r2t(R)
        assert r2t(R).dtype == "O"
        nt.assert_array_almost_equal(T[0:2, 2], np.r_[0, 0])
        nt.assert_array_almost_equal(T[:2, :2], R)

        theta = symbol("theta")
        R = rotx(theta)
        T = r2t(R)
        assert r2t(R).dtype == "O"
        nt.assert_array_almost_equal(T[0:3, 3], np.r_[0, 0, 0])
        # nt.assert_array_almost_equal(T[:3,:3], R)
        assert (T[:3, :3] == R).all()

    def test_t2r(self):
        # 3D
        t = [1, 2, 3]
        T = trotx(0.3, t=t)
        R = t2r(T)
        nt.assert_array_almost_equal(T[:3, :3], R)
        nt.assert_array_almost_equal(transl(T), np.array(t))

        # 2D
        t = [1, 2]
        T = trot2(0.3, t=t)
        R = t2r(T)
        nt.assert_array_almost_equal(T[:2, :2], R)
        nt.assert_array_almost_equal(transl2(T), np.array(t))

        with pytest.raises(ValueError):
            t2r(3)

        with pytest.raises(ValueError):
            r2t(np.eye(3, 4))

    def test_rt2tr(self):
        # 3D
        R = rotx(0.2)
        t = [3, 4, 5]
        T = rt2tr(R, t)
        nt.assert_array_almost_equal(t2r(T), R)
        nt.assert_array_almost_equal(transl(T), np.array(t))

        # 2D
        R = rot2(0.2)
        t = [3, 4]
        T = rt2tr(R, t)
        nt.assert_array_almost_equal(t2r(T), R)
        nt.assert_array_almost_equal(transl2(T), np.array(t))

        with pytest.raises(ValueError):
            rt2tr(3, 4)

        with pytest.raises(ValueError):
            rt2tr(np.eye(3, 4), [1, 2, 3, 4])

        with pytest.raises(ValueError):
            rt2tr(np.eye(4, 4), [1, 2, 3, 4])

        _ = rt2tr(np.ones((3, 3)), [1, 2, 3], check=False)
        with pytest.raises(ValueError):
            rt2tr(np.ones((3, 3)), [1, 2, 3], check=True)

    @pytest.mark.skipif(not _symbolics, reason="sympy required")
    def test_rt2tr_sym(self):
        theta = symbol("theta")
        R = rotx(theta)
        assert r2t(R).dtype == "O"

        theta = symbol("theta")
        R = rot2(theta)
        assert r2t(R).dtype == "O"

    def test_tr2rt(self):
        # 3D
        T = trotx(0.3, t=[1, 2, 3])
        R, t = tr2rt(T)
        nt.assert_array_almost_equal(T[:3, :3], R)
        nt.assert_array_almost_equal(T[:3, 3], t)

        # 2D
        T = trot2(0.3, t=[1, 2])
        R, t = tr2rt(T)
        nt.assert_array_almost_equal(T[:2, :2], R)
        nt.assert_array_almost_equal(T[:2, 2], t)

        with pytest.raises(ValueError):
            R, t = tr2rt(3)

        with pytest.raises(ValueError):
            R, t = tr2rt(np.eye(3, 4))

    def test_Ab2M(self):
        # 3D
        R = np.ones((3, 3))
        t = [3, 4, 5]
        T = Ab2M(R, t)
        nt.assert_array_almost_equal(T[:3, :3], R)
        nt.assert_array_almost_equal(T[:3, 3], np.array(t))
        nt.assert_array_almost_equal(T[3, :], np.array([0, 0, 0, 0]))

        # 2D
        R = np.ones((2, 2))
        t = [3, 4]
        T = Ab2M(R, t)
        nt.assert_array_almost_equal(T[:2, :2], R)
        nt.assert_array_almost_equal(T[:2, 2], np.array(t))
        nt.assert_array_almost_equal(T[2, :], np.array([0, 0, 0]))

        with pytest.raises(ValueError):
            Ab2M(3, 4)

        with pytest.raises(ValueError):
            Ab2M(np.eye(3, 4), [1, 2, 3, 4])

        with pytest.raises(ValueError):
            Ab2M(np.eye(4, 4), [1, 2, 3, 4])

    def test_checks(self):
        # 3D case, with rotation matrix
        R = np.eye(3)
        assert isR(R)
        assert not isrot2(R)
        assert isrot(R)
        assert not ishom(R)
        assert ishom2(R)
        assert not isrot2(R, True)
        assert isrot(R, True)
        assert not ishom(R, True)
        assert ishom2(R, True)

        # 3D case, invalid rotation matrix
        R = np.eye(3)
        R[0, 1] = 2
        assert not isR(R)
        assert not isrot2(R)
        assert isrot(R)
        assert not ishom(R)
        assert ishom2(R)
        assert not isrot2(R, True)
        assert not isrot(R, True)
        assert not ishom(R, True)
        assert not ishom2(R, True)

        # 3D case, with rotation matrix
        T = np.array([[1, 0, 0, 3], [0, 1, 0, 4], [0, 0, 1, 5], [0, 0, 0, 1]])
        assert not isR(T)
        assert not isrot2(T)
        assert not isrot(T)
        assert ishom(T)
        assert not ishom2(T)
        assert not isrot2(T, True)
        assert not isrot(T, True)
        assert ishom(T, True)
        assert not ishom2(T, True)

        # 3D case, invalid rotation matrix
        T = np.array([[1, 0, 0, 3], [0, 1, 1, 4], [0, 0, 1, 5], [0, 0, 0, 1]])
        assert not isR(T)
        assert not isrot2(T)
        assert not isrot(T)
        assert ishom(T)
        assert not ishom2(T)
        assert not isrot2(T, True)
        assert not isrot(T, True)
        assert not ishom(T, True)
        assert not ishom2(T, True)

        # 3D case, invalid bottom row
        T = np.array([[1, 0, 0, 3], [0, 1, 1, 4], [0, 0, 1, 5], [9, 0, 0, 1]])
        assert not isR(T)
        assert not isrot2(T)
        assert not isrot(T)
        assert ishom(T)
        assert not ishom2(T)
        assert not isrot2(T, True)
        assert not isrot(T, True)
        assert not ishom(T, True)
        assert not ishom2(T, True)

        # skew matrices
        S = np.array([[0, 2], [-2, 0]])
        nt.assert_equal(isskew(S), True)
        S[0, 0] = 1
        nt.assert_equal(isskew(S), False)

        S = np.array([[0, -3, 2], [3, 0, -1], [-2, 1, 0]])
        nt.assert_equal(isskew(S), True)
        S[0, 0] = 1
        nt.assert_equal(isskew(S), False)

    def test_homog(self):
        nt.assert_almost_equal(e2h([1, 2, 3]), np.c_[1, 2, 3, 1].T)

        nt.assert_almost_equal(h2e([2, 4, 6, 2]), np.c_[1, 2, 3].T)

    def test_homtrans(self):
        # 3D
        T = trotx(pi / 2, t=[1, 2, 3])
        v = [10, 12, 14]
        v2 = homtrans(T, v)
        nt.assert_almost_equal(v2, np.c_[11, -12, 15].T)
        v = np.c_[[10, 12, 14], [-3, -4, -5]]
        v2 = homtrans(T, v)
        nt.assert_almost_equal(v2, np.c_[[11, -12, 15], [-2, 7, -1]])

        # 2D
        T = trot2(pi / 2, t=[1, 2])
        v = [10, 12]
        v2 = homtrans(T, v)
        nt.assert_almost_equal(v2, np.c_[-11, 12].T)
        v = np.c_[[10, 12], [-3, -4]]
        v2 = homtrans(T, v)
        nt.assert_almost_equal(v2, np.c_[[-11, 12], [5, -1]])

        with pytest.raises(ValueError):
            T = trotx(pi / 2, t=[1, 2, 3])
            v = [10, 12]
            v2 = homtrans(T, v)

    def test_skew(self):
        # 3D
        sk = skew([1, 2, 3])
        assert sk.shape == (3, 3)
        nt.assert_almost_equal(sk + sk.T, np.zeros((3, 3)))
        assert sk[2, 1] == 1
        assert sk[0, 2] == 2
        assert sk[1, 0] == 3
        nt.assert_almost_equal(sk.diagonal(), np.r_[0, 0, 0])

        # 2D
        sk = skew([1])
        assert sk.shape == (2, 2)
        nt.assert_almost_equal(sk + sk.T, np.zeros((2, 2)))
        assert sk[1, 0] == 1
        nt.assert_almost_equal(sk.diagonal(), np.r_[0, 0])

        with pytest.raises(ValueError):
            sk = skew([1, 2])

    def test_vex(self):
        # 3D
        t = [3, 4, 5]
        sk = skew(t)
        nt.assert_almost_equal(vex(sk), t)

        # 2D
        t = [3]
        sk = skew(t)
        nt.assert_almost_equal(vex(sk), t)

        _ = vex(np.ones((3, 3)), check=False)
        with pytest.raises(ValueError):
            _ = vex(np.ones((3, 3)), check=True)

        with pytest.raises(ValueError):
            _ = vex(np.eye(4, 4))

    def test_isskew(self):
        t = [3, 4, 5]
        sk = skew(t)
        assert isskew(sk)
        sk[0, 0] = 3
        assert not isskew(sk)

        # 2D
        t = [3]
        sk = skew(t)
        assert isskew(sk)
        sk[0, 0] = 3
        assert not isskew(sk)

    def test_isskewa(self):
        # 3D
        t = [3, 4, 5, 6, 7, 8]
        sk = skewa(t)
        assert isskewa(sk)
        sk[0, 0] = 3
        assert not isskew(sk)
        sk = skewa(t)
        sk[3, 3] = 3
        assert not isskew(sk)

        # 2D
        t = [3, 4, 5]
        sk = skew(t)
        assert isskew(sk)
        sk[0, 0] = 3
        assert not isskew(sk)
        sk = skewa(t)
        sk[2, 2] = 3
        assert not isskew(sk)

    def test_skewa(self):
        # 3D
        sk = skewa([1, 2, 3, 4, 5, 6])
        assert sk.shape == (4, 4)
        nt.assert_almost_equal(sk.diagonal(), np.r_[0, 0, 0, 0])
        nt.assert_almost_equal(sk[-1, :], np.r_[0, 0, 0, 0])
        nt.assert_almost_equal(sk[:3, 3], [1, 2, 3])
        nt.assert_almost_equal(vex(sk[:3, :3]), [4, 5, 6])

        # 2D
        sk = skewa([1, 2, 3])
        assert sk.shape == (3, 3)
        nt.assert_almost_equal(sk.diagonal(), np.r_[0, 0, 0])
        nt.assert_almost_equal(sk[-1, :], np.r_[0, 0, 0])
        nt.assert_almost_equal(sk[:2, 2], [1, 2])
        nt.assert_almost_equal(vex(sk[:2, :2]), [3])

        with pytest.raises(ValueError):
            sk = skew([1, 2])

    def test_vexa(self):
        # 3D
        t = [1, 2, 3, 4, 5, 6]
        sk = skewa(t)
        nt.assert_almost_equal(vexa(sk), t)

        # 2D
        t = [1, 2, 3]
        sk = skewa(t)
        nt.assert_almost_equal(vexa(sk), t)

    def test_det(self):
        a = np.array([[1, 2], [3, 4]])
        assert np.linalg.det(a) == pytest.approx(det(a))

    @pytest.mark.skipif(not _symbolics, reason="sympy required")
    def test_det_sym(self):
        x, y = symbol("x y")
        a = np.array([[x, y], [y, x]])
        assert det(a) == x**2 - y**2
