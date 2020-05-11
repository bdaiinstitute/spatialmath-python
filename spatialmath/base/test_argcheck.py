#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 11:49:29 2020

@author: corkep
"""

import unittest
import numpy as np
import numpy.testing as nt

from spatialmath.base.argcheck import *

    
class Test_check(unittest.TestCase):
    
    def test_unit(self):
        
        nt.assert_equal(getunit(5, 'rad'), 5)
        nt.assert_equal(getunit(5, 'deg'), 5*math.pi/180.0)
        nt.assert_equal(getunit([3,4,5], 'rad'), [3,4,5])
        nt.assert_equal(getunit([3,4,5], 'deg'), [x*math.pi/180.0 for x in [3,4,5]])
        nt.assert_equal(getunit((3,4,5), 'rad'), [3,4,5])
        nt.assert_equal(getunit((3,4,5), 'deg'), [x*math.pi/180.0 for x in [3,4,5]])

        nt.assert_equal(getunit(np.array([3,4,5]), 'rad'), [3,4,5])
        nt.assert_equal(getunit(np.array([3,4,5]), 'deg'), [x*math.pi/180.0 for x in [3,4,5]])
        
    def test_isvector(self):
        # no length specified
        nt.assert_equal(isvector(2), True)
        nt.assert_equal(isvector(2.0), True)
        nt.assert_equal(isvector([1,2,3]), True)
        nt.assert_equal(isvector((1,2,3)), True)
        nt.assert_equal(isvector(np.array([1,2,3])), True)
        nt.assert_equal(isvector(np.array([[1,2,3]])), True)
        nt.assert_equal(isvector(np.array([[1],[2],[3]])), True)
        
        # length specified
        nt.assert_equal(isvector(2, 1), True)
        nt.assert_equal(isvector(2.0, 1), True)
        nt.assert_equal(isvector([1,2,3], 3), True)
        nt.assert_equal(isvector((1,2,3), 3), True)
        nt.assert_equal(isvector(np.array([1,2,3]), 3), True)
        nt.assert_equal(isvector(np.array([[1,2,3]]), 3), True)
        nt.assert_equal(isvector(np.array([[1],[2],[3]]), 3), True)
        
        # wrong length specified
        nt.assert_equal(isvector(2, 4), False)
        nt.assert_equal(isvector(2.0, 4), False)
        nt.assert_equal(isvector([1,2,3], 4), False)
        nt.assert_equal(isvector((1,2,3), 4), False)
        nt.assert_equal(isvector(np.array([1,2,3]), 4), False)
        nt.assert_equal(isvector(np.array([[1,2,3]]), 4), False)
        nt.assert_equal(isvector(np.array([[1],[2],[3]]), 4), False)
        
    def test_isvector(self):
        l = [1,2,3]
        nt.assert_raises(AssertionError, vector, l, 4 )
        
    def test_getvector(self):
        l = [1,2,3]
        t = (1,2,3)
        a = np.array(l)
        r = np.array([[1,2,3]])
        c = np.array([[1],[2],[3]])
        
        # input is list
        v = getvector(l)
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(len(v), 3)
        
        v = getvector(l, 3)
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(len(v), 3)
        
        v = getvector(l, out='sequence')
        self.assertIsInstance(v, list)
        nt.assert_equal(len(v), 3)
        
        v = getvector(l, 3, out='sequence')
        self.assertIsInstance(v, list)
        nt.assert_equal(len(v), 3)
        
        v = getvector(l, out='array')
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(v.shape, (3,))
        
        v = getvector(l, 3, out='array')
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(v.shape, (3,))
        
        v = getvector(l, out='row')
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(v.shape, (1,3))
        
        v = getvector(l, 3, out='row')
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(v.shape, (1,3))
        
        v = getvector(l, out='col')
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(v.shape, (3,1))
        
        v = getvector(l, 3, out='col')
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(v.shape, (3,1))
        
        nt.assert_raises(ValueError, getvector, l, 4 )
        nt.assert_raises(ValueError, getvector, l, 4, 'sequence' )
        nt.assert_raises(ValueError, getvector, l, 4, 'array' )
        nt.assert_raises(ValueError, getvector, l, 4, 'row' )
        nt.assert_raises(ValueError, getvector, l, 4, 'col' )
        
        # input is tuple
        
        v = getvector(t)
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(len(v), 3)
        
        v = getvector(t, 3)
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(len(v), 3)
        
        v = getvector(t, out='sequence')
        self.assertIsInstance(v, tuple)
        nt.assert_equal(len(v), 3)
        
        v = getvector(t, 3, out='sequence')
        self.assertIsInstance(v, tuple)
        nt.assert_equal(len(v), 3)
        
        v = getvector(t, out='array')
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(v.shape, (3,))
        
        v = getvector(t, 3, out='array')
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(v.shape, (3,))
        
        v = getvector(t, out='row')
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(v.shape, (1,3))
        
        v = getvector(t, 3, out='row')
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(v.shape, (1,3))
        
        v = getvector(t, out='col')
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(v.shape, (3,1))
        
        v = getvector(t, 3, out='col')
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(v.shape, (3,1))
        
        nt.assert_raises(ValueError, getvector, t, 4 )
        nt.assert_raises(ValueError, getvector, t, 4, 'sequence' )
        nt.assert_raises(ValueError, getvector, t, 4, 'array' )
        nt.assert_raises(ValueError, getvector, t, 4, 'row' )
        nt.assert_raises(ValueError, getvector, t, 4, 'col' )
        
        # input is array
        
        v = getvector(a)
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(len(v), 3)
        
        v = getvector(a, 3)
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(len(v), 3)
        
        v = getvector(a, out='sequence')
        self.assertIsInstance(v, list)
        nt.assert_equal(len(v), 3)
        
        v = getvector(a, 3, out='sequence')
        self.assertIsInstance(v, list)
        nt.assert_equal(len(v), 3)
        
        v = getvector(a, out='array')
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(v.shape, (3,))
        
        v = getvector(a, 3, out='array')
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(v.shape, (3,))
        
        v = getvector(a, out='row')
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(v.shape, (1,3))
        
        v = getvector(a, 3, out='row')
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(v.shape, (1,3))
        
        v = getvector(a, out='col')
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(v.shape, (3,1))
        
        v = getvector(a, 3, out='col')
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(v.shape, (3,1))
        
        nt.assert_raises(ValueError, getvector, a, 4 )
        nt.assert_raises(ValueError, getvector, a, 4, 'sequence' )
        nt.assert_raises(ValueError, getvector, a, 4, 'array' )
        nt.assert_raises(ValueError, getvector, a, 4, 'row' )
        nt.assert_raises(ValueError, getvector, a, 4, 'col' )

        # input is row
        
        v = getvector(r)
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(len(v), 3)
        
        v = getvector(r, 3)
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(len(v), 3)
        
        v = getvector(r, out='sequence')
        self.assertIsInstance(v, list)
        nt.assert_equal(len(v), 3)
        
        v = getvector(r, 3, out='sequence')
        self.assertIsInstance(v, list)
        nt.assert_equal(len(v), 3)
        
        v = getvector(r, out='array')
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(v.shape, (3,))
        
        v = getvector(r, 3, out='array')
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(v.shape, (3,))
        
        v = getvector(r, out='row')
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(v.shape, (1,3))
        
        v = getvector(r, 3, out='row')
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(v.shape, (1,3))
        
        v = getvector(r, out='col')
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(v.shape, (3,1))
        
        v = getvector(r, 3, out='col')
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(v.shape, (3,1))
        
        nt.assert_raises(ValueError, getvector, r, 4 )
        nt.assert_raises(ValueError, getvector, r, 4, 'sequence' )
        nt.assert_raises(ValueError, getvector, r, 4, 'array' )
        nt.assert_raises(ValueError, getvector, r, 4, 'row' )
        nt.assert_raises(ValueError, getvector, r, 4, 'col' )

        # input is col
        
        v = getvector(c)
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(len(v), 3)
        
        v = getvector(c, 3)
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(len(v), 3)
        
        v = getvector(c, out='sequence')
        self.assertIsInstance(v, list)
        nt.assert_equal(len(v), 3)
        
        v = getvector(c, 3, out='sequence')
        self.assertIsInstance(v, list)
        nt.assert_equal(len(v), 3)
        
        v = getvector(c, out='array')
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(v.shape, (3,))
        
        v = getvector(c, 3, out='array')
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(v.shape, (3,))
        
        v = getvector(c, out='row')
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(v.shape, (1,3))
        
        v = getvector(c, 3, out='row')
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(v.shape, (1,3))
        
        v = getvector(c, out='col')
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(v.shape, (3,1))
        
        v = getvector(c, 3, out='col')
        self.assertIsInstance(v, np.ndarray)
        nt.assert_equal(v.shape, (3,1))
        
        nt.assert_raises(ValueError, getvector, c, 4 )
        nt.assert_raises(ValueError, getvector, c, 4, 'sequence' )
        nt.assert_raises(ValueError, getvector, c, 4, 'array' )
        nt.assert_raises(ValueError, getvector, c, 4, 'row' )
        nt.assert_raises(ValueError, getvector, c, 4, 'col' )            
        

    
    def test_isnumberlist(self):
        nt.assert_equal(isnumberlist([1]), True)
        nt.assert_equal(isnumberlist([1,2]), True)
        nt.assert_equal(isnumberlist((1,)), True)
        nt.assert_equal(isnumberlist((1,2)), True)
        nt.assert_equal(isnumberlist(1), False)
        nt.assert_equal(isnumberlist([]), False)
        nt.assert_equal(isnumberlist(np.array([1,2,3])), False)
        
# ---------------------------------------------------------------------------------------#
if __name__ == '__main__':
    
    unittest.main()