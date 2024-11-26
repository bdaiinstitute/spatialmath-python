import numpy.testing as nt
import numpy as np

from spatialmath.spatialvector import *


class TestSpatialVector:
    def test_list_powers(self):
        x = SpatialVelocity.Empty()
        assert len(x) == 0
        x.append(SpatialVelocity([1, 2, 3, 4, 5, 6]))
        assert len(x) == 1

        x.append(SpatialVelocity([7, 8, 9, 10, 11, 12]))
        assert len(x) == 2

        y = x[0]
        assert isinstance(y, SpatialVelocity)
        assert len(y) == 1
        assert all(y.A == np.r_[1, 2, 3, 4, 5, 6])

        y = x[1]
        assert isinstance(y, SpatialVelocity)
        assert len(y) == 1
        assert all(y.A == np.r_[7, 8, 9, 10, 11, 12])

        x.insert(0, SpatialVelocity([20, 21, 22, 23, 24, 25]))

        y = x[0]
        assert isinstance(y, SpatialVelocity)
        assert len(y) == 1
        assert all(y.A == np.r_[20, 21, 22, 23, 24, 25])

        y = x[1]
        assert isinstance(y, SpatialVelocity)
        assert len(y) == 1
        assert all(y.A == np.r_[1, 2, 3, 4, 5, 6])

    def test_velocity(self):
        a = SpatialVelocity([1, 2, 3, 4, 5, 6])
        assert isinstance(a, SpatialVelocity)
        assert isinstance(a, SpatialVector)
        assert isinstance(a, SpatialM6)
        assert len(a) == 1
        assert all(a.A == np.r_[1, 2, 3, 4, 5, 6])

        a = SpatialVelocity(np.r_[1, 2, 3, 4, 5, 6])
        assert isinstance(a, SpatialVelocity)
        assert isinstance(a, SpatialVector)
        assert isinstance(a, SpatialM6)
        assert len(a) == 1
        assert all(a.A == np.r_[1, 2, 3, 4, 5, 6])

        s = str(a)
        assert isinstance(s, str)
        assert s.count("\n") == 0
        assert s.startswith("SpatialVelocity")

        r = np.random.rand(6, 10)
        a = SpatialVelocity(r)
        assert isinstance(a, SpatialVelocity)
        assert isinstance(a, SpatialVector)
        assert isinstance(a, SpatialM6)
        assert len(a) == 10

        b = a[3]
        assert isinstance(b, SpatialVelocity)
        assert isinstance(b, SpatialVector)
        assert isinstance(b, SpatialM6)
        assert len(b) == 1
        assert all(b.A == r[:, 3])

        s = str(a)
        assert isinstance(s, str)
        assert s.count("\n") == 9

    def test_acceleration(self):
        a = SpatialAcceleration([1, 2, 3, 4, 5, 6])
        assert isinstance(a, SpatialAcceleration)
        assert isinstance(a, SpatialVector)
        assert isinstance(a, SpatialM6)
        assert len(a) == 1
        assert all(a.A == np.r_[1, 2, 3, 4, 5, 6])

        a = SpatialAcceleration(np.r_[1, 2, 3, 4, 5, 6])
        assert isinstance(a, SpatialAcceleration)
        assert isinstance(a, SpatialVector)
        assert isinstance(a, SpatialM6)
        assert len(a) == 1
        assert all(a.A == np.r_[1, 2, 3, 4, 5, 6])

        s = str(a)
        assert isinstance(s, str)
        assert s.count("\n") == 0
        assert s.startswith("SpatialAcceleration")

        r = np.random.rand(6, 10)
        a = SpatialAcceleration(r)
        assert isinstance(a, SpatialAcceleration)
        assert isinstance(a, SpatialVector)
        assert isinstance(a, SpatialM6)
        assert len(a) == 10

        b = a[3]
        assert isinstance(b, SpatialAcceleration)
        assert isinstance(b, SpatialVector)
        assert isinstance(b, SpatialM6)
        assert len(b) == 1
        assert all(b.A == r[:, 3])

        s = str(a)
        assert isinstance(s, str)

    def test_force(self):
        a = SpatialForce([1, 2, 3, 4, 5, 6])
        assert isinstance(a, SpatialForce)
        assert isinstance(a, SpatialVector)
        assert isinstance(a, SpatialF6)
        assert len(a) == 1
        assert all(a.A == np.r_[1, 2, 3, 4, 5, 6])

        a = SpatialForce(np.r_[1, 2, 3, 4, 5, 6])
        assert isinstance(a, SpatialForce)
        assert isinstance(a, SpatialVector)
        assert isinstance(a, SpatialF6)
        assert len(a) == 1
        assert all(a.A == np.r_[1, 2, 3, 4, 5, 6])

        s = str(a)
        assert isinstance(s, str)
        assert s.count("\n") == 0
        assert s.startswith("SpatialForce")

        r = np.random.rand(6, 10)
        a = SpatialForce(r)
        assert isinstance(a, SpatialForce)
        assert isinstance(a, SpatialVector)
        assert isinstance(a, SpatialF6)
        assert len(a) == 10

        b = a[3]
        assert isinstance(b, SpatialForce)
        assert isinstance(b, SpatialVector)
        assert isinstance(b, SpatialF6)
        assert len(b) == 1
        assert all(b.A == r[:, 3])

        s = str(a)
        assert isinstance(s, str)

    def test_momentum(self):
        a = SpatialMomentum([1, 2, 3, 4, 5, 6])
        assert isinstance(a, SpatialMomentum)
        assert isinstance(a, SpatialVector)
        assert isinstance(a, SpatialF6)
        assert len(a) == 1
        assert all(a.A == np.r_[1, 2, 3, 4, 5, 6])

        a = SpatialMomentum(np.r_[1, 2, 3, 4, 5, 6])
        assert isinstance(a, SpatialMomentum)
        assert isinstance(a, SpatialVector)
        assert isinstance(a, SpatialF6)
        assert len(a) == 1
        assert all(a.A == np.r_[1, 2, 3, 4, 5, 6])

        s = str(a)
        assert isinstance(s, str)
        assert s.count("\n") == 0
        assert s.startswith("SpatialMomentum")

        r = np.random.rand(6, 10)
        a = SpatialMomentum(r)
        assert isinstance(a, SpatialMomentum)
        assert isinstance(a, SpatialVector)
        assert isinstance(a, SpatialF6)
        assert len(a) == 10

        b = a[3]
        assert isinstance(b, SpatialMomentum)
        assert isinstance(b, SpatialVector)
        assert isinstance(b, SpatialF6)
        assert len(b) == 1
        assert all(b.A == r[:, 3])

        s = str(a)
        assert isinstance(s, str)

    def test_arith(self):
        # just test SpatialVelocity since all types derive from same superclass

        r1 = np.r_[1, 2, 3, 4, 5, 6]
        r2 = np.r_[7, 8, 9, 10, 11, 12]
        a1 = SpatialVelocity(r1)
        a2 = SpatialVelocity(r2)

        assert all((a1 + a2).A == r1 + r2)
        assert all((a1 - a2).A == r1 - r2)
        assert all((-a1).A == -r1)

    def test_inertia(self):
        # constructor
        i0 = SpatialInertia()
        nt.assert_equal(i0.A, np.zeros((6, 6)))

        i1 = SpatialInertia(np.eye(6, 6))
        nt.assert_equal(i1.A, np.eye(6, 6))

        i2 = SpatialInertia(m=1, r=(1, 2, 3))
        nt.assert_almost_equal(i2.A, i2.A.T)

        i3 = SpatialInertia(m=1, r=(1, 2, 3), I=np.ones((3, 3)))
        nt.assert_almost_equal(i3.A, i3.A.T)

        # addition
        m_a, m_b = 1.1, 2.2
        r = (1, 2, 3)
        i4a, i4b = SpatialInertia(m=m_a, r=r), SpatialInertia(m=m_b, r=r)
        nt.assert_almost_equal((i4a + i4b).A, SpatialInertia(m=m_a + m_b, r=r).A)

        # isvalid - note this method is very barebone, to be improved
        assert SpatialInertia().isvalid(np.ones((6, 6)), check=False)

    def test_products(self):
        # v x v = a  *, v x F6 = a
        # a x I, I x a
        # v x I, I x v
        # twist x v, twist x a, twist x F
        pass
