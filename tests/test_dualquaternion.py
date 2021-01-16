import math
from math import pi
import numpy as np

import numpy.testing as nt
import unittest

from spatialmath import DualQuaternion, UnitDualQuaternion, Quaternion, SE3
from spatialmath import base


def qcompare(x, y):
    if isinstance(x, Quaternion):
        x = x.vec
    elif isinstance(x, SMPose):
        x = x.A
    if isinstance(y, Quaternion):
        y = y.vec
    elif isinstance(y, SMPose):
        y = y.A
    nt.assert_array_almost_equal(x, y)

class TestDualQuaternion(unittest.TestCase):

    def test_init(self):

        dq = DualQuaternion(Quaternion([1.,2,3,4]), Quaternion([5.,6,7,8]))
        nt.assert_array_almost_equal(dq.vec, np.r_[1,2,3,4,5,6,7,8])

        dq = DualQuaternion([1.,2,3,4,5,6,7,8])
        nt.assert_array_almost_equal(dq.vec, np.r_[1,2,3,4,5,6,7,8])
        dq = DualQuaternion(np.r_[1,2,3,4,5,6,7,8])
        nt.assert_array_almost_equal(dq.vec, np.r_[1,2,3,4,5,6,7,8])

    def test_pure(self):

        dq = DualQuaternion.Pure([1.,2,3])
        nt.assert_array_almost_equal(dq.vec, np.r_[1,0,0,0,  0,1,2,3])

    def test_strings(self):

        dq = DualQuaternion(Quaternion([1.,2,3,4]), Quaternion([5.,6,7,8]))
        self.assertIsInstance(str(dq), str)
        self.assertIsInstance(repr(dq), str)

    def test_conj(self):
        dq = DualQuaternion(Quaternion([1.,2,3,4]), Quaternion([5.,6,7,8]))
        nt.assert_array_almost_equal(dq.conj().vec, np.r_[1,-2,-3,-4, 5,-6,-7,-8])

    # def test_norm(self):
    #     q1 = Quaternion([1.,2,3,4])
    #     q2 = Quaternion([5.,6,7,8])

    #     dq = DualQuaternion(q1, q2)
    #     nt.assert_array_almost_equal(dq.norm(), (q1.norm(), q2.norm()))

    def test_plus(self):
        dq = DualQuaternion(Quaternion([1.,2,3,4]), Quaternion([5.,6,7,8]))
        s = dq + dq
        nt.assert_array_almost_equal(s.vec, 2*np.r_[1,2,3,4,5,6,7,8])

    def test_minus(self):
        dq = DualQuaternion(Quaternion([1.,2,3,4]), Quaternion([5.,6,7,8]))
        s = dq - dq
        nt.assert_array_almost_equal(s.vec, np.zeros((8,)))

    def test_matrix(self):

        dq1 = DualQuaternion(Quaternion([1.,2,3,4]), Quaternion([5.,6,7,8]))

        M = dq1.matrix()
        self.assertIsInstance(M, np.ndarray)
        self.assertEqual(M.shape, (8,8))

    def test_multiply(self):
        dq1 = DualQuaternion(Quaternion([1.,2,3,4]), Quaternion([5.,6,7,8]))
        dq2 = DualQuaternion(Quaternion([4,3,2,1]), Quaternion([5,6,7,8]))

        M = dq1.matrix()
        v = dq2.vec
        nt.assert_array_almost_equal(M @ v, (dq1 * dq2).vec)

    def test_unit(self):
        pass


class TestUnitDualQuaternion(unittest.TestCase):

    def test_init(self):

        T = SE3.Rx(pi/4)
        dq = UnitDualQuaternion(T)
        nt.assert_array_almost_equal(dq.SE3().A, T.A)

    def test_norm(self):
        T = SE3.Rx(pi/4)
        dq = UnitDualQuaternion(T)
        nt.assert_array_almost_equal(dq.norm(), (1,0))

    def test_multiply(self):
        T1 = SE3.Rx(pi/4)
        T2 = SE3.Rz(-pi/3)

        T = T1 * T2

        d1 = UnitDualQuaternion(T1)
        d2 = UnitDualQuaternion(T2)

        d = d1 * d2
        nt.assert_array_almost_equal(d.SE3().A, T.A)


# ---------------------------------------------------------------------------------------#
if __name__ == '__main__':  # pragma: no cover

    unittest.main()
