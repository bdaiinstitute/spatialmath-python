#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 11:49:29 2020

@author: corkep
"""

import numpy as np
import numpy.testing as nt
import pytest
from spatialmath.base.argcheck import *


class Test_check:
    def test_ismatrix(self):
        a = np.eye(3, 3)
        assert ismatrix(a, (3, 3))
        assert not ismatrix(a, (4, 3))
        assert not ismatrix(a, (3, 4))
        assert not ismatrix(a, (4, 4))

        assert ismatrix(a, (-1, 3))
        assert ismatrix(a, (3, -1))
        assert ismatrix(a, (-1, -1))

        assert not ismatrix(1, (-1, -1))

    def test_assertmatrix(self):
        with pytest.raises(TypeError):
            assertmatrix(3)
        with pytest.raises(TypeError):
            assertmatrix("not a matrix")

        with pytest.raises(TypeError):
            a = np.eye(3, 3, dtype=complex)
            assertmatrix(a)

        a = np.eye(3, 3)

        assertmatrix(a)
        assertmatrix(a, (3, 3))
        assertmatrix(a, (None, 3))
        assertmatrix(a, (3, None))

        with pytest.raises(ValueError):
            assertmatrix(a, (4, 3))
        with pytest.raises(ValueError):
            assertmatrix(a, (4, None))
        with pytest.raises(ValueError):
            assertmatrix(a, (None, 4))

    def test_getmatrix(self):
        a = np.random.rand(4, 3)
        assert getmatrix(a, (4, 3)).shape == (4, 3)
        assert getmatrix(a, (None, 3)).shape == (4, 3)
        assert getmatrix(a, (4, None)).shape == (4, 3)
        assert getmatrix(a, (None, None)).shape == (4, 3)
        with pytest.raises(ValueError):
            m = getmatrix(a, (5, 3))
        with pytest.raises(ValueError):
            m = getmatrix(a, (5, None))
        with pytest.raises(ValueError):
            m = getmatrix(a, (None, 4))

        with pytest.raises(TypeError):
            m = getmatrix({}, (4, 3))

        a = np.r_[1, 2, 3, 4]
        assert getmatrix(a, (1, 4)).shape == (1, 4)
        assert getmatrix(a, (4, 1)).shape == (4, 1)
        assert getmatrix(a, (2, 2)).shape == (2, 2)
        with pytest.raises(ValueError):
            m = getmatrix(a, (5, None))
        with pytest.raises(ValueError):
            m = getmatrix(a, (None, 5))

        a = [1, 2, 3, 4]
        assert getmatrix(a, (1, 4)).shape == (1, 4)
        assert getmatrix(a, (4, 1)).shape == (4, 1)
        assert getmatrix(a, (2, 2)).shape == (2, 2)
        with pytest.raises(ValueError):
            m = getmatrix(a, (5, None))
        with pytest.raises(ValueError):
            m = getmatrix(a, (None, 5))

        a = 7
        assert getmatrix(a, (1, 1)).shape == (1, 1)
        assert getmatrix(a, (None, None)).shape == (1, 1)
        with pytest.raises(ValueError):
            m = getmatrix(a, (2, 1))
        with pytest.raises(ValueError):
            m = getmatrix(a, (1, 2))
        with pytest.raises(ValueError):
            m = getmatrix(a, (None, 2))
        with pytest.raises(ValueError):
            m = getmatrix(a, (2, None))

        a = 7.0
        assert getmatrix(a, (1, 1)).shape == (1, 1)
        assert getmatrix(a, (None, None)).shape == (1, 1)
        with pytest.raises(ValueError):
            m = getmatrix(a, (2, 1))
        with pytest.raises(ValueError):
            m = getmatrix(a, (1, 2))
        with pytest.raises(ValueError):
            m = getmatrix(a, (None, 2))
        with pytest.raises(ValueError):
            m = getmatrix(a, (2, None))

    def test_verifymatrix(self):
        with pytest.raises(TypeError):
            assertmatrix(3)
        with pytest.raises(TypeError):
            verifymatrix([3, 4])

        a = np.eye(3, 3)

        verifymatrix(a, (3, 3))
        with pytest.raises(ValueError):
            verifymatrix(a, (3, 4))

    def test_unit(self):
        assert isinstance(getunit(1), np.ndarray)
        assert isinstance(getunit([1, 2]), np.ndarray)
        assert isinstance(getunit((1, 2)), np.ndarray)
        assert isinstance(getunit(np.r_[1, 2]), np.ndarray)
        assert isinstance(getunit(1.0, dim=0), float)

        nt.assert_equal(getunit(5, "rad"), 5)
        nt.assert_equal(getunit(5, "deg"), 5 * math.pi / 180.0)
        nt.assert_equal(getunit([3, 4, 5], "rad"), [3, 4, 5])
        nt.assert_almost_equal(
            getunit([3, 4, 5], "deg"), [x * math.pi / 180.0 for x in [3, 4, 5]]
        )
        nt.assert_equal(getunit((3, 4, 5), "rad"), [3, 4, 5])
        nt.assert_almost_equal(
            getunit((3, 4, 5), "deg"),
            np.array([x * math.pi / 180.0 for x in [3, 4, 5]]),
        )

        nt.assert_equal(getunit(np.array([3, 4, 5]), "rad"), [3, 4, 5])
        nt.assert_almost_equal(
            getunit(np.array([3, 4, 5]), "deg"),
            [x * math.pi / 180.0 for x in [3, 4, 5]],
        )

    def test_isvector(self):
        # no length specified
        assert isvector(2)
        assert isvector(2.0)
        assert isvector([1, 2, 3])
        assert isvector((1, 2, 3))
        assert isvector(np.array([1, 2, 3]))
        assert isvector(np.array([[1, 2, 3]]))
        assert isvector(np.array([[1], [2], [3]]))

        # length specified
        assert isvector(2, 1)
        assert isvector(2.0, 1)
        assert isvector([1, 2, 3], 3)
        assert isvector((1, 2, 3), 3)
        assert isvector(np.array([1, 2, 3]), 3)
        assert isvector(np.array([[1, 2, 3]]), 3)
        assert isvector(np.array([[1], [2], [3]]), 3)

        # wrong length specified
        assert not isvector(2, 4)
        assert not isvector(2.0, 4)
        assert not isvector([1, 2, 3], 4)
        assert not isvector((1, 2, 3), 4)
        assert not isvector(np.array([1, 2, 3]), 4)
        assert not isvector(np.array([[1, 2, 3]]), 4)
        assert not isvector(np.array([[1], [2], [3]]), 4)

    def test_isvector(self):
        l = [1, 2, 3]
        nt.assert_raises(ValueError, assertvector, l, 4)

    def test_getvector(self):
        l = [1, 2, 3]
        t = (1, 2, 3)
        a = np.array(l)
        r = np.array([[1, 2, 3]])
        c = np.array([[1], [2], [3]])

        # input is list
        v = getvector(l)
        assert isinstance(v, np.ndarray)
        nt.assert_equal(len(v), 3)

        v = getvector(l, 3)
        assert isinstance(v, np.ndarray)
        nt.assert_equal(len(v), 3)

        v = getvector(l, out="sequence")
        assert isinstance(v, list)
        nt.assert_equal(len(v), 3)

        v = getvector(l, 3, out="sequence")
        assert isinstance(v, list)
        nt.assert_equal(len(v), 3)

        v = getvector(l, out="array")
        assert isinstance(v, np.ndarray)
        nt.assert_equal(v.shape, (3,))

        v = getvector(l, 3, out="array")
        assert isinstance(v, np.ndarray)
        nt.assert_equal(v.shape, (3,))

        v = getvector(l, out="row")
        assert isinstance(v, np.ndarray)
        nt.assert_equal(v.shape, (1, 3))

        v = getvector(l, 3, out="row")
        assert isinstance(v, np.ndarray)
        nt.assert_equal(v.shape, (1, 3))

        v = getvector(l, out="col")
        assert isinstance(v, np.ndarray)
        nt.assert_equal(v.shape, (3, 1))

        v = getvector(l, 3, out="col")
        assert isinstance(v, np.ndarray)
        nt.assert_equal(v.shape, (3, 1))

        nt.assert_raises(ValueError, getvector, l, 4)
        nt.assert_raises(ValueError, getvector, l, 4, "sequence")
        nt.assert_raises(ValueError, getvector, l, 4, "array")
        nt.assert_raises(ValueError, getvector, l, 4, "row")
        nt.assert_raises(ValueError, getvector, l, 4, "col")

        # input is tuple

        v = getvector(t)
        assert isinstance(v, np.ndarray)
        nt.assert_equal(len(v), 3)

        v = getvector(t, 3)
        assert isinstance(v, np.ndarray)
        nt.assert_equal(len(v), 3)

        v = getvector(t, out="sequence")
        assert isinstance(v, tuple)
        nt.assert_equal(len(v), 3)

        v = getvector(t, 3, out="sequence")
        assert isinstance(v, tuple)
        nt.assert_equal(len(v), 3)

        v = getvector(t, out="array")
        assert isinstance(v, np.ndarray)
        nt.assert_equal(v.shape, (3,))

        v = getvector(t, 3, out="array")
        assert isinstance(v, np.ndarray)
        nt.assert_equal(v.shape, (3,))

        v = getvector(t, out="row")
        assert isinstance(v, np.ndarray)
        nt.assert_equal(v.shape, (1, 3))

        v = getvector(t, 3, out="row")
        assert isinstance(v, np.ndarray)
        nt.assert_equal(v.shape, (1, 3))

        v = getvector(t, out="col")
        assert isinstance(v, np.ndarray)
        nt.assert_equal(v.shape, (3, 1))

        v = getvector(t, 3, out="col")
        assert isinstance(v, np.ndarray)
        nt.assert_equal(v.shape, (3, 1))

        nt.assert_raises(ValueError, getvector, t, 4)
        nt.assert_raises(ValueError, getvector, t, 4, "sequence")
        nt.assert_raises(ValueError, getvector, t, 4, "array")
        nt.assert_raises(ValueError, getvector, t, 4, "row")
        nt.assert_raises(ValueError, getvector, t, 4, "col")

        # input is array

        v = getvector(a)
        assert isinstance(v, np.ndarray)
        nt.assert_equal(len(v), 3)

        v = getvector(a, 3)
        assert isinstance(v, np.ndarray)
        nt.assert_equal(len(v), 3)

        v = getvector(a, out="sequence")
        assert isinstance(v, list)
        nt.assert_equal(len(v), 3)

        v = getvector(a, 3, out="sequence")
        assert isinstance(v, list)
        nt.assert_equal(len(v), 3)

        v = getvector(a, out="array")
        assert isinstance(v, np.ndarray)
        nt.assert_equal(v.shape, (3,))

        v = getvector(a, 3, out="array")
        assert isinstance(v, np.ndarray)
        nt.assert_equal(v.shape, (3,))

        v = getvector(a, out="row")
        assert isinstance(v, np.ndarray)
        nt.assert_equal(v.shape, (1, 3))

        v = getvector(a, 3, out="row")
        assert isinstance(v, np.ndarray)
        nt.assert_equal(v.shape, (1, 3))

        v = getvector(a, out="col")
        assert isinstance(v, np.ndarray)
        nt.assert_equal(v.shape, (3, 1))

        v = getvector(a, 3, out="col")
        assert isinstance(v, np.ndarray)
        nt.assert_equal(v.shape, (3, 1))

        nt.assert_raises(ValueError, getvector, a, 4)
        nt.assert_raises(ValueError, getvector, a, 4, "sequence")
        nt.assert_raises(ValueError, getvector, a, 4, "array")
        nt.assert_raises(ValueError, getvector, a, 4, "row")
        nt.assert_raises(ValueError, getvector, a, 4, "col")

        # input is row

        v = getvector(r)
        assert isinstance(v, np.ndarray)
        nt.assert_equal(len(v), 3)

        v = getvector(r, 3)
        assert isinstance(v, np.ndarray)
        nt.assert_equal(len(v), 3)

        v = getvector(r, out="sequence")
        assert isinstance(v, list)
        nt.assert_equal(len(v), 3)

        v = getvector(r, 3, out="sequence")
        assert isinstance(v, list)
        nt.assert_equal(len(v), 3)

        v = getvector(r, out="array")
        assert isinstance(v, np.ndarray)
        nt.assert_equal(v.shape, (3,))

        v = getvector(r, 3, out="array")
        assert isinstance(v, np.ndarray)
        nt.assert_equal(v.shape, (3,))

        v = getvector(r, out="row")
        assert isinstance(v, np.ndarray)
        nt.assert_equal(v.shape, (1, 3))

        v = getvector(r, 3, out="row")
        assert isinstance(v, np.ndarray)
        nt.assert_equal(v.shape, (1, 3))

        v = getvector(r, out="col")
        assert isinstance(v, np.ndarray)
        nt.assert_equal(v.shape, (3, 1))

        v = getvector(r, 3, out="col")
        assert isinstance(v, np.ndarray)
        nt.assert_equal(v.shape, (3, 1))

        nt.assert_raises(ValueError, getvector, r, 4)
        nt.assert_raises(ValueError, getvector, r, 4, "sequence")
        nt.assert_raises(ValueError, getvector, r, 4, "array")
        nt.assert_raises(ValueError, getvector, r, 4, "row")
        nt.assert_raises(ValueError, getvector, r, 4, "col")

        # input is col

        v = getvector(c)
        assert isinstance(v, np.ndarray)
        nt.assert_equal(len(v), 3)

        v = getvector(c, 3)
        assert isinstance(v, np.ndarray)
        nt.assert_equal(len(v), 3)

        v = getvector(c, out="sequence")
        assert isinstance(v, list)
        nt.assert_equal(len(v), 3)

        v = getvector(c, 3, out="sequence")
        assert isinstance(v, list)
        nt.assert_equal(len(v), 3)

        v = getvector(c, out="array")
        assert isinstance(v, np.ndarray)
        nt.assert_equal(v.shape, (3,))

        v = getvector(c, 3, out="array")
        assert isinstance(v, np.ndarray)
        nt.assert_equal(v.shape, (3,))

        v = getvector(c, out="row")
        assert isinstance(v, np.ndarray)
        nt.assert_equal(v.shape, (1, 3))

        v = getvector(c, 3, out="row")
        assert isinstance(v, np.ndarray)
        nt.assert_equal(v.shape, (1, 3))

        v = getvector(c, out="col")
        assert isinstance(v, np.ndarray)
        nt.assert_equal(v.shape, (3, 1))

        v = getvector(c, 3, out="col")
        assert isinstance(v, np.ndarray)
        nt.assert_equal(v.shape, (3, 1))

        nt.assert_raises(ValueError, getvector, c, 4)
        nt.assert_raises(ValueError, getvector, c, 4, "sequence")
        nt.assert_raises(ValueError, getvector, c, 4, "array")
        nt.assert_raises(ValueError, getvector, c, 4, "row")
        nt.assert_raises(ValueError, getvector, c, 4, "col")

    def test_isnumberlist(self):
        nt.assert_equal(isnumberlist([1]), True)
        nt.assert_equal(isnumberlist([1, 2]), True)
        nt.assert_equal(isnumberlist((1,)), True)
        nt.assert_equal(isnumberlist((1, 2)), True)
        nt.assert_equal(isnumberlist(1), False)
        nt.assert_equal(isnumberlist([]), False)
        nt.assert_equal(isnumberlist(np.array([1, 2, 3])), False)

    def test_isvectorlist(self):
        a = [np.r_[1, 2], np.r_[3, 4], np.r_[5, 6]]
        assert isvectorlist(a, 2)

        a = [(1, 2), (3, 4), (5, 6)]
        assert not isvectorlist(a, 2)

        a = [np.r_[1, 2], np.r_[3, 4], np.r_[5, 6, 7]]
        assert not isvectorlist(a, 2)

    def test_islistof(self):
        a = [3, 4, 5]
        assert islistof(a, int)
        assert not islistof(a, float)
        assert islistof(a, lambda x: isinstance(x, int))

        assert islistof(a, int, 3)
        assert not islistof(a, int, 2)

        a = [3, 4.5, 5.6]
        assert not islistof(a, int)
        assert islistof(a, (int, float))
        a = [[1, 2], [3, 4], [5, 6]]
        assert islistof(a, lambda x: islistof(x, int, 2))
