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
from spatialmath import SE3, SO3, SE2, UnitQuaternion
import numpy as np
from spatialmath.base import *
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


class TestSO3:
    @classmethod
    def tearDownClass(cls):
        plt.close("all")

    def test_constructor(self):
        # construct from matrix
        R = SO3(rotx(0.2))
        nt.assert_equal(len(R), 1)
        array_compare(R, rotx(0.2))
        assert isinstance(R, SO3)

        # construct from canonic rotation
        R = SO3.Rx(0.2)
        nt.assert_equal(len(R), 1)
        array_compare(R, rotx(0.2))
        assert isinstance(R, SO3)

        R = SO3.Ry(0.2)
        nt.assert_equal(len(R), 1)
        array_compare(R, roty(0.2))
        assert isinstance(R, SO3)

        R = SO3.Rz(0.2)
        nt.assert_equal(len(R), 1)
        array_compare(R, rotz(0.2))
        assert isinstance(R, SO3)

        # OA
        R = SO3.OA([0, 1, 0], [0, 0, 1])
        nt.assert_equal(len(R), 1)
        array_compare(R, np.eye(3))
        assert isinstance(R, SO3)

        np.random.seed(32)
        # random
        R = SO3.Rand()
        nt.assert_equal(len(R), 1)
        assert isinstance(R, SO3)

        # random constrained
        R = SO3.Rand(theta_range=(0.1, 0.7))
        assert isinstance(R, SO3)
        assert R.A.shape == (3, 3)
        assert R.angvec()[0] <= 0.7
        assert R.angvec()[0] >= 0.1

        # copy constructor
        R = SO3.Rx(pi / 2)
        R2 = SO3(R)
        R = SO3.Ry(pi / 2)
        array_compare(R2, rotx(pi / 2))

    def test_constructor_Eul(self):
        R = SO3.Eul([0.1, 0.2, 0.3])
        nt.assert_equal(len(R), 1)
        array_compare(R, eul2r([0.1, 0.2, 0.3]))
        assert isinstance(R, SO3)

        R = SO3.Eul(0.1, 0.2, 0.3)
        nt.assert_equal(len(R), 1)
        array_compare(R, eul2r([0.1, 0.2, 0.3]))
        assert isinstance(R, SO3)

        R = SO3.Eul(np.r_[0.1, 0.2, 0.3])
        nt.assert_equal(len(R), 1)
        array_compare(R, eul2r([0.1, 0.2, 0.3]))
        assert isinstance(R, SO3)

        R = SO3.Eul([10, 20, 30], unit="deg")
        nt.assert_equal(len(R), 1)
        array_compare(R, eul2r([10, 20, 30], unit="deg"))
        assert isinstance(R, SO3)

        R = SO3.Eul(10, 20, 30, unit="deg")
        nt.assert_equal(len(R), 1)
        array_compare(R, eul2r([10, 20, 30], unit="deg"))
        assert isinstance(R, SO3)

    def test_constructor_RPY(self):
        R = SO3.RPY(0.1, 0.2, 0.3, order="zyx")
        nt.assert_equal(len(R), 1)
        array_compare(R, rpy2r([0.1, 0.2, 0.3], order="zyx"))
        assert isinstance(R, SO3)

        R = SO3.RPY(10, 20, 30, unit="deg", order="zyx")
        nt.assert_equal(len(R), 1)
        array_compare(R, rpy2r([10, 20, 30], order="zyx", unit="deg"))
        assert isinstance(R, SO3)

        R = SO3.RPY([0.1, 0.2, 0.3], order="zyx")
        nt.assert_equal(len(R), 1)
        array_compare(R, rpy2r([0.1, 0.2, 0.3], order="zyx"))
        assert isinstance(R, SO3)

        R = SO3.RPY(np.r_[0.1, 0.2, 0.3], order="zyx")
        nt.assert_equal(len(R), 1)
        array_compare(R, rpy2r([0.1, 0.2, 0.3], order="zyx"))
        assert isinstance(R, SO3)

        # check default
        R = SO3.RPY([0.1, 0.2, 0.3])
        nt.assert_equal(len(R), 1)
        array_compare(R, rpy2r([0.1, 0.2, 0.3], order="zyx"))
        assert isinstance(R, SO3)

        # XYZ order

        R = SO3.RPY(0.1, 0.2, 0.3, order="xyz")
        nt.assert_equal(len(R), 1)
        array_compare(R, rpy2r([0.1, 0.2, 0.3], order="xyz"))
        assert isinstance(R, SO3)

        R = SO3.RPY(10, 20, 30, unit="deg", order="xyz")
        nt.assert_equal(len(R), 1)
        array_compare(R, rpy2r([10, 20, 30], order="xyz", unit="deg"))
        assert isinstance(R, SO3)

        R = SO3.RPY([0.1, 0.2, 0.3], order="xyz")
        nt.assert_equal(len(R), 1)
        array_compare(R, rpy2r([0.1, 0.2, 0.3], order="xyz"))
        assert isinstance(R, SO3)

        R = SO3.RPY(np.r_[0.1, 0.2, 0.3], order="xyz")
        nt.assert_equal(len(R), 1)
        array_compare(R, rpy2r([0.1, 0.2, 0.3], order="xyz"))
        assert isinstance(R, SO3)

    def test_constructor_AngVec(self):
        # angvec
        R = SO3.AngVec(0.2, [1, 0, 0])
        nt.assert_equal(len(R), 1)
        array_compare(R, rotx(0.2))
        assert isinstance(R, SO3)

        R = SO3.AngVec(0.3, [0, 1, 0])
        nt.assert_equal(len(R), 1)
        array_compare(R, roty(0.3))
        assert isinstance(R, SO3)

    def test_constructor_TwoVec(self):
        # Randomly selected vectors
        v1 = [1, 73, -42]
        v2 = [0, 0.02, 57]
        v3 = [-2, 3, 9]

        # x and y given
        R = SO3.TwoVectors(x=v1, y=v2)
        assert isinstance(R, SO3)
        nt.assert_almost_equal(R.det(), 1, 5)
        # x axis should equal normalized x vector
        nt.assert_almost_equal(R.R[:, 0], v1 / np.linalg.norm(v1), 5)

        # y and z given
        R = SO3.TwoVectors(y=v2, z=v3)
        assert isinstance(R, SO3)
        nt.assert_almost_equal(R.det(), 1, 5)
        # y axis should equal normalized y vector
        nt.assert_almost_equal(R.R[:, 1], v2 / np.linalg.norm(v2), 5)

        # x and z given
        R = SO3.TwoVectors(x=v3, z=v1)
        assert isinstance(R, SO3)
        nt.assert_almost_equal(R.det(), 1, 5)
        # x axis should equal normalized x vector
        nt.assert_almost_equal(R.R[:, 0], v3 / np.linalg.norm(v3), 5)

    def test_conversion(self):
        R = SO3.AngleAxis(0.7, [1,2,3])
        q = UnitQuaternion([11,7,3,-6])

        R_from_q = SO3(q.R)
        q_from_R = UnitQuaternion(R)

        nt.assert_array_almost_equal(R.UnitQuaternion(), q_from_R)
        nt.assert_array_almost_equal(R.UnitQuaternion().SO3(), R)

        nt.assert_array_almost_equal(q.SO3(), R_from_q)
        nt.assert_array_almost_equal(q.SO3().UnitQuaternion(), q)


    def test_shape(self):
        a = SO3.identity()
        assert a._A.shape == a.shape

    def test_about(self):
        R = SO3.identity()
        R.about

    def test_str(self):
        R = SO3.identity()

        s = str(R)
        assert isinstance(s, str)
        assert s.count("\n") == 3

        s = repr(R)
        assert isinstance(s, str)
        assert s.count("\n") == 2

    def test_printline(self):
        R = SO3.Rx(0.3)

        R.printline()
        # s = R.printline(file=None)
        # assert isinstance(s, str)

        R = SO3.Rx([0.3, 0.4, 0.5])
        s = R.printline(file=None)
        # assert isinstance(s, str)
        # assert s.count('\n') == 2

    @pytest.mark.skipif(
        sys.platform.startswith("darwin") and sys.version_info < (3, 11),
        reason="tkinter bug with mac",
    )
    def test_plot(self):
        plt.close("all")

        R = SO3.Rx(0.3)
        R.plot(block=False)

        R2 = SO3.Rx(0.6)
        # R.animate()
        # R.animate(start=R.inv())

    def test_tests(self):
        R = SO3.identity()

        assert R.isrot() == True
        assert R.isrot2() == False
        assert R.ishom() == False
        assert R.ishom2() == False

    def test_properties(self):
        R = SO3.identity()

        assert R.isSO == True
        assert R.isSE == False

        array_compare(R.n, np.r_[1, 0, 0])
        array_compare(R.n, np.r_[1, 0, 0])
        array_compare(R.n, np.r_[1, 0, 0])

        nt.assert_equal(R.N, 3)
        nt.assert_equal(R.shape, (3, 3))

        R = SO3.Rx(0.3)
        array_compare(R.inv() * R, np.eye(3, 3))

    def test_arith(self):
        R = SO3.identity()

        # sum
        a = R + R
        assert not isinstance(a, SO3)
        array_compare(a, np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]]))

        a = R + 1
        assert not isinstance(a, SO3)
        array_compare(a, np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]]))

        # a = 1 + R
        # assert not isinstance(a, SO3)
        # array_compare(a, np.array([ [2,1,1], [1,2,1], [1,1,2]]))

        a = R + np.eye(3)
        assert not isinstance(a, SO3)
        array_compare(a, np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]]))

        # a =  np.eye(3) + R
        # assert not isinstance(a, SO3)
        # array_compare(a, np.array([ [2,0,0], [0,2,0], [0,0,2]]))
        #  this invokes the __add__ method for numpy

        # difference
        R = SO3.identity()

        a = R - R
        assert not isinstance(a, SO3)
        array_compare(a, np.zeros((3, 3)))

        a = R - 1
        assert not isinstance(a, SO3)
        array_compare(a, np.array([[0, -1, -1], [-1, 0, -1], [-1, -1, 0]]))

        # a = 1 - R
        # assert not isinstance(a, SO3)
        # array_compare(a, -np.array([ [0,-1,-1], [-1,0,-1], [-1,-1,0]]))

        a = R - np.eye(3)
        assert not isinstance(a, SO3)
        array_compare(a, np.zeros((3, 3)))

        # a =  np.eye(3) - R
        # assert not isinstance(a, SO3)
        # array_compare(a, np.zeros((3,3)))

        # multiply
        R = SO3.identity()

        a = R * R
        assert isinstance(a, SO3)
        array_compare(a, R)

        a = R * 2
        assert not isinstance(a, SO3)
        array_compare(a, 2 * np.eye(3))

        a = 2 * R
        assert not isinstance(a, SO3)
        array_compare(a, 2 * np.eye(3))

        R = SO3.identity()
        R *= SO3.Rx(pi / 2)
        assert isinstance(R, SO3)
        array_compare(R, rotx(pi / 2))

        R = SO3.identity()
        R *= 2
        assert not isinstance(R, SO3)
        array_compare(R, 2 * np.eye(3))

        array_compare(SO3.Rx(pi / 2) * SO3.Ry(pi / 2) * SO3.Rx(-pi / 2), SO3.Rz(pi / 2))

        array_compare(SO3.Ry(pi / 2) * [1, 0, 0], np.c_[0, 0, -1].T)

        # SO3 x vector
        vx = np.r_[1, 0, 0]
        vy = np.r_[0, 1, 0]
        vz = np.r_[0, 0, 1]

        def cv(v):
            return np.c_[v]

        nt.assert_equal(isinstance(SO3.Rx(pi / 2) * vx, np.ndarray), True)
        print(vx)
        print(SO3.Rx(pi / 2) * vx)
        print(cv(vx))
        array_compare(SO3.Rx(pi / 2) * vx, cv(vx))
        array_compare(SO3.Rx(pi / 2) * vy, cv(vz))
        array_compare(SO3.Rx(pi / 2) * vz, cv(-vy))

        array_compare(SO3.Ry(pi / 2) * vx, cv(-vz))
        array_compare(SO3.Ry(pi / 2) * vy, cv(vy))
        array_compare(SO3.Ry(pi / 2) * vz, cv(vx))

        array_compare(SO3.Rz(pi / 2) * vx, cv(vy))
        array_compare(SO3.Rz(pi / 2) * vy, cv(-vx))
        array_compare(SO3.Rz(pi / 2) * vz, cv(vz))

        # divide
        R = SO3.Ry(0.3)
        a = R / R
        assert isinstance(a, SO3)
        array_compare(a, np.eye(3))

        a = R / 2
        assert not isinstance(a, SO3)
        array_compare(a, roty(0.3) / 2)

        # power

        R = SO3.Rx(pi / 2)
        R = R**2
        array_compare(R, SO3.Rx(pi))

        R = SO3.Rx(pi / 2)
        R **= 2
        array_compare(R, SO3.Rx(pi))

        R = SO3.Rx(pi / 4)
        R = R ** (-2)
        array_compare(R, SO3.Rx(-pi / 2))

        R = SO3.Rx(pi / 4)
        R **= -2
        array_compare(R, SO3.Rx(-pi / 2))

    def test_functions(self):
        # inv
        # .T

        # conversion to SE2
        poseSE3 = SE3.Tx(3.3) * SE3.Rz(1.5)
        poseSE2 = poseSE3.yaw_SE2()
        nt.assert_almost_equal(poseSE3.R[0:2, 0:2], poseSE2.R[0:2, 0:2])
        nt.assert_equal(poseSE3.x, poseSE2.x)
        nt.assert_equal(poseSE3.y, poseSE2.y)

    def test_functions_lie(self):
        R = SO3.EulerVec([0.42, 0.73, -1.17])

        # Check log and exponential map
        nt.assert_equal(R, SO3.Exp(R.log()))
        np.testing.assert_equal((R.inv() * R).log(), np.zeros([3, 3]))

        # Check euler vector map
        nt.assert_equal(R, SO3.EulerVec(R.eulervec()))
        np.testing.assert_equal((R.inv() * R).eulervec(), np.zeros(3))

    def test_identity(self):
        nt.assert_equal(SO3.identity().A, np.eye(3))

# ============================== SE3 =====================================#


class TestSE3:
    @classmethod
    def tearDownClass(cls):
        plt.close("all")

    def test_constructor(self):
        # construct from matrix
        R = SE3(trotx(0.2))
        nt.assert_equal(len(R), 1)
        array_compare(R, trotx(0.2))
        assert isinstance(R, SE3)

        # construct from canonic rotation
        R = SE3.Rx(0.2)
        nt.assert_equal(len(R), 1)
        array_compare(R, trotx(0.2))
        assert isinstance(R, SE3)

        R = SE3.Ry(0.2)
        nt.assert_equal(len(R), 1)
        array_compare(R, troty(0.2))
        assert isinstance(R, SE3)

        R = SE3.Rz(0.2)
        nt.assert_equal(len(R), 1)
        array_compare(R, trotz(0.2))
        assert isinstance(R, SE3)

        # construct from canonic translation
        R = SE3.Tx(0.2)
        nt.assert_equal(len(R), 1)
        array_compare(R, transl(0.2, 0, 0))
        assert isinstance(R, SE3)

        R = SE3.Ty(0.2)
        nt.assert_equal(len(R), 1)
        array_compare(R, transl(0, 0.2, 0))
        assert isinstance(R, SE3)

        R = SE3.Tz(0.2)
        nt.assert_equal(len(R), 1)
        array_compare(R, transl(0, 0, 0.2))
        assert isinstance(R, SE3)

        # triple angle
        R = SE3.Eul([0.1, 0.2, 0.3])
        nt.assert_equal(len(R), 1)
        array_compare(R, eul2tr([0.1, 0.2, 0.3]))
        assert isinstance(R, SE3)

        R = SE3.Eul(np.r_[0.1, 0.2, 0.3])
        nt.assert_equal(len(R), 1)
        array_compare(R, eul2tr([0.1, 0.2, 0.3]))
        assert isinstance(R, SE3)

        R = SE3.Eul([10, 20, 30], unit="deg")
        nt.assert_equal(len(R), 1)
        array_compare(R, eul2tr([10, 20, 30], unit="deg"))
        assert isinstance(R, SE3)

        R = SE3.RPY([0.1, 0.2, 0.3])
        nt.assert_equal(len(R), 1)
        array_compare(R, rpy2tr([0.1, 0.2, 0.3]))
        assert isinstance(R, SE3)

        R = SE3.RPY(np.r_[0.1, 0.2, 0.3])
        nt.assert_equal(len(R), 1)
        array_compare(R, rpy2tr([0.1, 0.2, 0.3]))
        assert isinstance(R, SE3)

        R = SE3.RPY([10, 20, 30], unit="deg")
        nt.assert_equal(len(R), 1)
        array_compare(R, rpy2tr([10, 20, 30], unit="deg"))
        assert isinstance(R, SE3)

        R = SE3.RPY([0.1, 0.2, 0.3], order="xyz")
        nt.assert_equal(len(R), 1)
        array_compare(R, rpy2tr([0.1, 0.2, 0.3], order="xyz"))
        assert isinstance(R, SE3)

        # angvec
        R = SE3.AngVec(0.2, [1, 0, 0])
        nt.assert_equal(len(R), 1)
        array_compare(R, trotx(0.2))
        assert isinstance(R, SE3)

        R = SE3.AngVec(0.3, [0, 1, 0])
        nt.assert_equal(len(R), 1)
        array_compare(R, troty(0.3))
        assert isinstance(R, SE3)

        # OA
        R = SE3.OA([0, 1, 0], [0, 0, 1])
        nt.assert_equal(len(R), 1)
        array_compare(R, np.eye(4))
        assert isinstance(R, SE3)

        np.random.seed(65)
        # random
        R = SE3.Rand()
        nt.assert_equal(len(R), 1)
        assert isinstance(R, SE3)

        # random 
        T = SE3.Rand()
        R = T.R
        t = T.t
        T = SE3.Rt(R, t)
        assert isinstance(T, SE3)
        assert T.A.shape == (4, 4)

        nt.assert_equal(T.R, R)
        nt.assert_equal(T.t, t)

        nt.assert_equal(T.x, t[0])
        nt.assert_equal(T.y, t[1])
        nt.assert_equal(T.z, t[2])

        # random constrained
        T = SE3.Rand(theta_range=(0.1, 0.7))
        assert isinstance(T, SE3)
        assert T.A.shape == (4, 4)
        assert T.angvec()[0] <= 0.7
        assert T.angvec()[0] >= 0.1

        # copy constructor
        R = SE3.Rx(pi / 2)
        R2 = SE3(R)
        R = SE3.Ry(pi / 2)
        array_compare(R2, trotx(pi / 2))

        # SO3
        T = SE3(SO3.identity())
        nt.assert_equal(len(T), 1)
        assert isinstance(T, SE3)
        nt.assert_equal(T.A, np.eye(4))

        # SE2
        T = SE3(SE2(1, 2, 0.4))
        nt.assert_equal(len(T), 1)
        assert isinstance(T, SE3)
        assert T.A.shape == (4, 4)
        nt.assert_equal(T.t, [1, 2, 0])

        # Bad number of arguments
        with pytest.raises(ValueError):
            T = SE3(1.0, 0.0)
        with pytest.raises(TypeError):
            T = SE3(1.0, 0.0, 0.0, 0.0)

    def test_shape(self):
        a = SE3.identity()
        assert a._A.shape == a.shape

    def test_tests(self):
        R = SE3.identity()

        assert R.isrot() == False
        assert R.isrot2() == False
        assert R.ishom() == True
        assert R.ishom2() == False

    def test_properties(self):
        R = SE3.identity()

        assert R.isSO == False
        assert R.isSE == True

        array_compare(R.n, np.r_[1, 0, 0])
        array_compare(R.n, np.r_[1, 0, 0])
        array_compare(R.n, np.r_[1, 0, 0])

        nt.assert_equal(R.N, 3)
        nt.assert_equal(R.shape, (4, 4))

        # Testing the CopyFrom function
        mutable_array = np.eye(4)
        pass_by_ref = SE3(mutable_array)
        pass_by_val = SE3.CopyFrom(mutable_array)
        mutable_array[0, 3] = 5.0
        nt.assert_allclose(pass_by_val.data[0], np.eye(4))
        nt.assert_allclose(pass_by_ref.data[0], mutable_array)
        nt.assert_raises(
            AssertionError, nt.assert_allclose, pass_by_val.data[0], pass_by_ref.data[0]
        )

    def test_arith(self):
        T = SE3(1, 2, 3)

        # sum
        a = T + T
        assert not isinstance(a, SE3)
        array_compare(
            a, np.array([[2, 0, 0, 2], [0, 2, 0, 4], [0, 0, 2, 6], [0, 0, 0, 2]])
        )

        a = T + 1
        assert not isinstance(a, SE3)
        array_compare(
            a, np.array([[2, 1, 1, 2], [1, 2, 1, 3], [1, 1, 2, 4], [1, 1, 1, 2]])
        )

        # a = 1 + T
        # assert not isinstance(a, SE3)
        # array_compare(a, np.array([ [2,1,1], [1,2,1], [1,1,2]]))

        a = T + np.eye(4)
        assert not isinstance(a, SE3)
        array_compare(
            a, np.array([[2, 0, 0, 1], [0, 2, 0, 2], [0, 0, 2, 3], [0, 0, 0, 2]])
        )

        # a =  np.eye(3) + T
        # assert not isinstance(a, SE3)
        # array_compare(a, np.array([ [2,0,0], [0,2,0], [0,0,2]]))
        #  this invokes the __add__ method for numpy

        # difference
        T = SE3(1, 2, 3)

        a = T - T
        assert not isinstance(a, SE3)
        array_compare(a, np.zeros((4, 4)))

        a = T - 1
        assert not isinstance(a, SE3)
        array_compare(
            a,
            np.array([[0, -1, -1, 0], [-1, 0, -1, 1], [-1, -1, 0, 2], [-1, -1, -1, 0]]),
        )

        # a = 1 - T
        # assert not isinstance(a, SE3)
        # array_compare(a, -np.array([ [0,-1,-1], [-1,0,-1], [-1,-1,0]]))

        a = T - np.eye(4)
        assert not isinstance(a, SE3)
        array_compare(
            a, np.array([[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 3], [0, 0, 0, 0]])
        )

        # a =  np.eye(3) - T
        # assert not isinstance(a, SE3)
        # array_compare(a, np.zeros((3,3)))

        a = T
        a -= T
        assert not isinstance(a, SE3)
        array_compare(a, np.zeros((4, 4)))

        # multiply
        T = SE3(1, 2, 3)

        a = T * T
        assert isinstance(a, SE3)
        array_compare(a, transl(2, 4, 6))

        a = T * 2
        assert not isinstance(a, SE3)
        array_compare(a, 2 * transl(1, 2, 3))

        a = 2 * T
        assert not isinstance(a, SE3)
        array_compare(a, 2 * transl(1, 2, 3))

        T = SE3(1, 2, 3)
        T *= SE3.Ry(pi / 2)
        assert isinstance(T, SE3)
        array_compare(
            T, np.array([[0, 0, 1, 1], [0, 1, 0, 2], [-1, 0, 0, 3], [0, 0, 0, 1]])
        )

        T = SE3.identity()
        T *= 2
        assert not isinstance(T, SE3)
        array_compare(T, 2 * np.eye(4))

        array_compare(SE3.Rx(pi / 2) * SE3.Ry(pi / 2) * SE3.Rx(-pi / 2), SE3.Rz(pi / 2))

        array_compare(SE3.Ry(pi / 2) * [1, 0, 0], np.c_[0, 0, -1].T)

        # SE3 x vector
        vx = np.r_[1, 0, 0]
        vy = np.r_[0, 1, 0]
        vz = np.r_[0, 0, 1]

        def cv(v):
            return np.c_[v]

        nt.assert_equal(isinstance(SE3.Tx(pi / 2) * vx, np.ndarray), True)
        array_compare(SE3.Rx(pi / 2) * vx, cv(vx))
        array_compare(SE3.Rx(pi / 2) * vy, cv(vz))
        array_compare(SE3.Rx(pi / 2) * vz, cv(-vy))

        array_compare(SE3.Ry(pi / 2) * vx, cv(-vz))
        array_compare(SE3.Ry(pi / 2) * vy, cv(vy))
        array_compare(SE3.Ry(pi / 2) * vz, cv(vx))

        array_compare(SE3.Rz(pi / 2) * vx, cv(vy))
        array_compare(SE3.Rz(pi / 2) * vy, cv(-vx))
        array_compare(SE3.Rz(pi / 2) * vz, cv(vz))

        # divide
        T = SE3.Ry(0.3)
        a = T / T
        assert isinstance(a, SE3)
        array_compare(a, np.eye(4))

        a = T / 2
        assert not isinstance(a, SE3)
        array_compare(a, troty(0.3) / 2)

    def test_angle(self):
        # angle between SO3's
        r1 = SO3.Rx(0.1)
        r2 = SO3.Rx(0.2)
        for metric in range(6):
            assert r1.angdist(other=r1, metric=metric) == pytest.approx(0.0)
            assert r1.angdist(other=r2, metric=metric) > 0.0
            assert r1.angdist(other=r2, metric=metric) == pytest.approx(r2.angdist(other=r1, metric=metric))
        # angle between SE3's
        p1a, p1b = SE3.Rx(0.1), SE3.Rx(0.1, t=(1, 2, 3))
        p2a, p2b = SE3.Rx(0.2), SE3.Rx(0.2, t=(3, 2, 1))
        for metric in range(6):
            assert p1a.angdist(other=p1a, metric=metric) == pytest.approx(0.0)
            assert p1a.angdist(other=p2a, metric=metric) > 0.0
            assert p1a.angdist(other=p1b, metric=metric) == pytest.approx(0.0)
            assert p1a.angdist(other=p2a, metric=metric) == pytest.approx(p2a.angdist(other=p1a, metric=metric))
            assert p1a.angdist(other=p2a, metric=metric) == pytest.approx(p1a.angdist(other=p2b, metric=metric))
        # angdist is not implemented for mismatched types
        with pytest.raises(ValueError):
            _ = r1.angdist(p1a)

        with pytest.raises(ValueError):
            _ = r1._op2(right=p1a, op=r1.angdist)

        with pytest.raises(ValueError):
            _ = p1a._op2(right=r1, op=p1a.angdist)

        # in general, the _op2 interface enforces an isinstance check.
        with pytest.raises(TypeError):
            _ = r1._op2(right=(1, 0, 0), op=r1.angdist)

    def test_functions(self):
        # inv
        # .T
        pass

    def test_functions_vect(self):
        # inv
        # .T
        pass

    def test_identity(self):
        nt.assert_equal(SE3.identity().A, np.eye(4))
