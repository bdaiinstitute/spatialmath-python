import numpy.testing as nt
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt
import sys
import pytest

"""
we will assume that the primitives rotx,trotx, etc. all work
"""
from math import pi
from spatialmath.pose2d import *

# from spatialmath import super_pose as sp
from spatialmath.base import *
import spatialmath.base.argcheck as argcheck
import spatialmath as sm
from spatialmath.baseposematrix import BasePoseMatrix
from spatialmath.twist import BaseTwist


def array_compare(x, y):
    if isinstance(x, BasePoseMatrix):
        x = x.A
    if isinstance(y, BasePoseMatrix):
        y = y.A
    if isinstance(x, BaseTwist):
        x = x.S
    if isinstance(y, BaseTwist):
        y = y.S
    nt.assert_array_almost_equal(x, y)


class TestSO2:
    @classmethod
    def tearDownClass(cls):
        plt.close("all")

    def test_constructor(self):
        # null case
        x = SO2()
        assert isinstance(x, SO2)
        assert len(x) == 1
        array_compare(x.A, np.eye(2, 2))

        ## from angle

        array_compare(SO2(0).A, np.eye(2))
        array_compare(SO2(pi / 2).A, rot2(pi / 2))
        array_compare(SO2(90, unit="deg").A, rot2(pi / 2))

        ## from R

        array_compare(SO2(np.eye(2, 2)).A, np.eye(2, 2))

        array_compare(SO2(rot2(pi / 2)).A, rot2(pi / 2))
        array_compare(SO2(rot2(pi)).A, rot2(pi))

        ## R,T
        array_compare(SO2(np.eye(2)).R, np.eye(2))

        array_compare(SO2(rot2(pi / 2)).R, rot2(pi / 2))

        # TODO assert SO2(R).R == R

        ## copy constructor
        r = SO2(0.3)
        c = SO2(r)
        array_compare(r, c)
        r = SO2(0.4)
        array_compare(c, SO2(0.3))

    def test_primitive_convert(self):
        # char

        s = str(SO2())
        assert isinstance(s, str)

    def test_shape(self):
        a = SO2()
        assert a._A.shape == a.shape

    def test_constructor_Exp(self):
        array_compare(SO2.Exp(skew(0.3)).R, rot2(0.3))
        array_compare(SO2.Exp(0.3).R, rot2(0.3))

    def test_isa(self):
        assert SO2.isvalid(rot2(0))

        assert not SO2.isvalid(1)

    def test_resulttype(self):
        r = SO2()
        assert isinstance(r, SO2)

        assert isinstance(r * r, SO2)

        assert isinstance(r / r, SO2)

        assert isinstance(r.inv(), SO2)

    @pytest.mark.parametrize(
        'left, right, expected',
        [
            (SO2(0), SO2(), SO2(0)),
            (SO2(), SO2(0), SO2(0)),
            (SO2(pi/2), np.r_[1, 0], np.c_[np.r_[0, 1]]),
        ],
    )
    def test_multiply(self, left, right, expected):
        array_compare(left * right, expected)

    @pytest.mark.parametrize(
        'left, right, expected',
        [
            (SO2(pi/2), SO2(), SO2(pi/2)),
            (SO2(pi/2), SO2(pi/2), SO2()),
        ],
    )
    def test_divide(self, left, right, expected):
        array_compare(left / right, expected)

    def test_conversions(self):
        T = SO2(pi / 2).SE2()

        assert isinstance(T, SE2)

        ## Lie stuff
        th = 0.3
        RR = SO2(th)
        array_compare(RR.log(), skew(th))

    def test_miscellany(self):
        r = SO2(
            0.3,
        )
        assert np.linalg.det(r.A) == pytest.approx(1)

        assert r.N == 2

        assert not r.isSE

    def test_printline(self):
        R = SO2(0.3)

        R.printline()
        # s = R.printline(file=None)
        # assert isinstance(s, str)

    @pytest.mark.skipif(
        sys.platform.startswith("darwin") and sys.version_info < (3, 11),
        reason="tkinter bug with mac",
    )
    def test_plot(self):
        plt.close("all")

        R = SO2(0.3)
        R.plot(block=False)

        R2 = SO2(0.6)
        # R.animate()
        # R.animate(start=R2)

    def test_identity(self):
        array_compare(SO2.identity().A, np.eye(2, 2))

# ============================== SE2 =====================================#


class TestSE2:
    @classmethod
    def tearDownClass(cls):
        plt.close("all")

    def test_constructor(self):
        assert isinstance(SE2(), SE2)

        ## null
        array_compare(SE2().A, np.eye(3, 3))

        # from x,y
        x = SE2(2, 3)
        assert isinstance(x, SE2)
        assert len(x) == 1
        array_compare(x.A, np.array([[1, 0, 2], [0, 1, 3], [0, 0, 1]]))

        x = SE2([2, 3])
        assert isinstance(x, SE2)
        assert len(x) == 1
        array_compare(x.A, np.array([[1, 0, 2], [0, 1, 3], [0, 0, 1]]))

        # from x,y,theta
        x = SE2(2, 3, pi / 2)
        assert isinstance(x, SE2)
        assert len(x) == 1
        array_compare(x.A, np.array([[0, -1, 2], [1, 0, 3], [0, 0, 1]]))

        x = SE2([2, 3, pi / 2])
        assert isinstance(x, SE2)
        assert len(x) == 1
        array_compare(x.A, np.array([[0, -1, 2], [1, 0, 3], [0, 0, 1]]))

        x = SE2(2, 3, 90, unit="deg")
        assert isinstance(x, SE2)
        assert len(x) == 1
        array_compare(x.A, np.array([[0, -1, 2], [1, 0, 3], [0, 0, 1]]))

        x = SE2([2, 3, 90], unit="deg")
        assert isinstance(x, SE2)
        assert len(x) == 1
        array_compare(x.A, np.array([[0, -1, 2], [1, 0, 3], [0, 0, 1]]))

        ## T
        T = transl2(1, 2) @ trot2(0.3)
        x = SE2(T)
        assert isinstance(x, SE2)
        assert len(x) == 1
        array_compare(x.A, T)

        ## copy constructor
        TT = SE2(x)
        array_compare(SE2(TT).A, T)
        x = SE2()
        array_compare(SE2(TT).A, T)

    def test_shape(self):
        a = SE2()
        assert a._A.shape == a.shape

    def test_constructor_Exp(self):
        array_compare(SE2.Exp(skewa([1, 2, 0])), transl2(1, 2))
        array_compare(SE2.Exp(np.r_[1, 2, 0]), transl2(1, 2))

    def test_isa(self):
        assert SE2.isvalid(trot2(0))
        assert not SE2.isvalid(1)

    def test_resulttype(self):
        t = SE2()
        assert isinstance(t, SE2)
        assert isinstance(t * t, SE2)
        assert isinstance(t / t, SE2)
        assert isinstance(t.inv(), SE2)
        assert isinstance(t + t, np.ndarray)
        assert isinstance(t + 1, np.ndarray)
        assert isinstance(t - 1, np.ndarray)
        assert isinstance(1 + t, np.ndarray)
        assert isinstance(1 - t, np.ndarray)
        assert isinstance(2 * t, np.ndarray)
        assert isinstance(t * 2, np.ndarray)

    def test_inverse(self):
        T1 = transl2(1, 2) @ trot2(0.3)
        TT1 = SE2(T1)

        # test inverse
        array_compare(TT1.inv().A, np.linalg.inv(T1))

        array_compare(TT1 * TT1.inv(), np.eye(3))
        array_compare(TT1.inv() * TT1, np.eye(3))

    def test_Rt(self):
        TT1 = SE2.Rand()
        T1 = TT1.A
        R1 = t2r(T1)
        t1 = transl2(T1)

        array_compare(TT1.A, T1)
        array_compare(TT1.R, R1)
        array_compare(TT1.t, t1)
        assert TT1.x == t1[0]
        assert TT1.y == t1[1]

    def test_arith(self):
        TT1 = SE2.Rand()
        T1 = TT1.A
        TT2 = SE2.Rand()
        T2 = TT2.A

        I = SE2()

        ## SE2, * SE2, product
        # scalar x scalar

        array_compare(TT1 * TT2, T1 @ T2)
        array_compare(TT2 * TT1, T2 @ T1)
        array_compare(TT1 * I, T1)
        array_compare(TT2 * I, TT2)

        ## SE2, * vector product
        vx = np.r_[1, 0]
        vy = np.r_[0, 1]

        # scalar x scalar

        array_compare(TT1 * vy, h2e(T1 @ e2h(vy)))

    def test_defs(self):
        # log
        # x = SE2.Exp([2, 3, 0.5])
        # array_compare(x.log(), np.array([[0, -0.5, 2], [0.5, 0, 3], [0, 0, 0]]))
        pass

    def test_conversions(self):
        ##  SE2,                     convert to SE2, class

        TT = SE2(1, 2, 0.3)

        array_compare(TT, transl2(1, 2) @ trot2(0.3))

        ## xyt
        array_compare(TT.xyt(), np.r_[1, 2, 0.3])

        ## Lie stuff
        x = TT.log()
        assert isskewa(x)

    def test_interp(self):
        TT = SE2(2, -4, 0.6)
        I = SE2()

        z = I.interp(TT, s=0)[0]
        assert isinstance(z, SE2)

        array_compare(I.interp(TT, s=0)[0], I)
        array_compare(I.interp(TT, s=1)[0], TT)
        array_compare(I.interp(TT, s=0.5)[0], SE2(1, -2, 0.3))

        R1 = SO2(math.pi - 0.1)
        R2 = SO2(-math.pi + 0.2)
        array_compare(R1.interp(R2, s=0.5, shortest=False)[0], SO2(0.05))
        array_compare(R1.interp(R2, s=0.5, shortest=True)[0], SO2(-math.pi + 0.05))

        T1 = SE2(0, 0, math.pi - 0.1)
        T2 = SE2(0, 0, -math.pi + 0.2)
        array_compare(T1.interp(T2, s=0.5, shortest=False)[0], SE2(0, 0, 0.05))
        array_compare(T1.interp(T2, s=0.5, shortest=True)[0], SE2(0, 0, -math.pi + 0.05))

    def test_miscellany(self):
        TT = SE2(1, 2, 0.3)

        assert TT.A.shape == (3, 3)

        assert TT.isSE

        assert isinstance(TT, SE2)

    def test_display(self):
        T1 = SE2.Rand()

        T1.printline()

    @pytest.mark.skipif(
        sys.platform.startswith("darwin") and sys.version_info < (3, 11),
        reason="tkinter bug with mac",
    )
    def test_graphics(self):
        plt.close("all")
        T1 = SE2.Rand()
        T2 = SE2.Rand()

        T1.plot(block=False, dims=[-2, 2])

        T1.animate(repeat=False, dims=[-2, 2], nframes=10)
        T1.animate(T0=T2, repeat=False, dims=[-2, 2], nframes=10)
