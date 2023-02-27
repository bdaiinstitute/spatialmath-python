#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 14:19:04 2020

@author: corkep

"""


import numpy as np
import numpy.testing as nt
import unittest
from math import pi
import math
from scipy.linalg import logm, expm

from spatialmath.base.transforms3d import *
from spatialmath.base.transformsNd import isR, t2r, r2t, rt2tr

import matplotlib.pyplot as plt


class Test3D(unittest.TestCase):
    def test_plot(self):
        plt.figure()
        # test options
        trplot(
            transl(1, 2, 3),
            block=False,
            frame="A",
            style="line",
            width=1,
            dims=[0, 10, 0, 10, 0, 10],
        )
        trplot(
            transl(1, 2, 3),
            block=False,
            frame="A",
            style="arrow",
            width=1,
            dims=[0, 10, 0, 10, 0, 10],
        )
        trplot(
            transl(1, 2, 3),
            block=False,
            frame="A",
            style="rgb",
            width=1,
            dims=[0, 10, 0, 10, 0, 10],
        )
        trplot(transl(3, 1, 2), block=False, color="red", width=3, frame="B")
        trplot(
            transl(4, 3, 1) @ trotx(math.pi / 3),
            block=False,
            color="green",
            frame="c",
            dims=[0, 4, 0, 4, 0, 4],
        )

        # test for iterable
        plt.clf()
        T = [transl(1, 2, 3), transl(2, 3, 4), transl(3, 4, 5)]
        trplot(T)

        plt.close("all")

    def test_animate(self):
        tranimate(transl(1, 2, 3), repeat=False, wait=True)

        tranimate(transl(1, 2, 3), repeat=False, wait=True)
        # run again, with axes already created
        tranimate(transl(1, 2, 3), repeat=False, wait=True, dims=[0, 10, 0, 10, 0, 10])

        plt.close("all")
        # test animate with line not arrow, text, test with SO(3)


# ---------------------------------------------------------------------------------------#
if __name__ == "__main__":
    unittest.main()
