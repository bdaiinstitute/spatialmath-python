import numpy.testing as nt
import unittest

"""
we will assume that the primitives rotx,trotx, etc. all work
"""
from math import pi
from spatialmath.pose3d import *
from spatialmath import super_pose as sp
from spatialmath.base import *
import spatialmath.base.argcheck as argcheck

def array_compare(x, y):
    if isinstance(x, sp.SuperPose):
        x = x.A
    if isinstance(y, sp.SuperPose):
        y = y.A
    nt.assert_array_almost_equal(x, y)
                           
                           
class TestSO2(unittest.TestCase):
    pass

class TestSE2(unittest.TestCase):
    pass



        
# ---------------------------------------------------------------------------------------#
if __name__ == '__main__':

    
    unittest.main()
        
