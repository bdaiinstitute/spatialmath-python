import numpy.testing as nt
import unittest

"""
we will assume that the primitives rotx,trotx, etc. all work
"""
from spatialmath import SE3, SO3, SE2
import numpy as np
import spatialmath.base.transforms3d as t3d


class TestTransforms3D(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        pass

    def test_tr2angvec(self):
        true_ang = 1.51
        true_vec = np.array([0., 1., 0.])
        eps = 1e-08

        # show that tr2angvec works on true rotation matrix
        R = SO3.Ry(true_ang)
        ang, vec = t3d.tr2angvec(R.A, check=True)
        nt.assert_equal(ang, true_ang)
        nt.assert_equal(vec, true_vec)

        # check a rotation matrix that should fail
        badR = SO3.Ry(true_ang).A[:, :] + eps
        with self.assertRaises(ValueError):
            t3d.tr2angvec(badR, check=True)

        # run without check
        ang, vec = t3d.tr2angvec(badR, check=False)
        nt.assert_almost_equal(ang, true_ang)
        nt.assert_equal(vec, true_vec)
        

# ---------------------------------------------------------------------------------------#
if __name__ == '__main__':
    
    unittest.main()
