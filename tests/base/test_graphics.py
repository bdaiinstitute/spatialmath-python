import unittest
import numpy as np
from spatialmath.base import *

# test graphics primitives
# TODO check they actually create artists


class TestGraphics(unittest.TestCase):
    def test_plotvol2(self):
        plotvol2(5)

    def test_plotvol3(self):
        plotvol3(5)

    def test_plot_point(self):
        plot_point((2, 3))
        plot_point(np.r_[2, 3])
        plot_point((2, 3), "x")
        plot_point((2, 3), "x", text="foo")

    def test_plot_text(self):

        plot_text((2, 3), "foo")
        plot_text(np.r_[2, 3], "foo")

    def test_plot_box(self):
        plot_box("r--", centre=(-2, -3), wh=(1, 1))
        plot_box(lt=(1, 1), rb=(2, 0), filled=True, color="b")
        plot_box(lrbt=(1, 2, 0, 1), filled=True, color="b")
        plot_box(ltrb=(1, 0, 2, 0), filled=True, color="b")
        plot_box(lt=(1, 2), wh=(2, 3))
        plot_box(lbwh=(1, 2, 3, 4))
        plot_box(centre=(1, 2), wh=(2, 3))

    def test_plot_circle(self):
        plot_circle(1, (0, 0), "r")  # red circle
        plot_circle(2, (0, 0), "b--")  # blue dashed circle
        plot_circle(0.5, (0, 0), filled=True, color="y")  # yellow filled circle

    def test_ellipse(self):
        plot_ellipse(np.diag((1, 2)), "r")  # red ellipse
        plot_ellipse(np.diag((1, 2)), "b--")  # blue dashed ellipse
        plot_ellipse(
            np.diag((1, 2)), centre=(1, 1), filled=True, color="y"
        )  # yellow filled ellipse

    def test_plot_homline(self):
        plot_homline((1, 2, 3))
        plot_homline((2, 1, 3))
        plot_homline((1, -2, 3), "k--")

    def test_cuboid(self):
        plot_cuboid((1, 2, 3), color="g")
        plot_cuboid((1, 2, 3), centre=(2, 3, 4), color="g")
        plot_cuboid((1, 2, 3), filled=True, color="y")

    def test_sphere(self):
        plot_sphere(0.3, color="r")
        plot_sphere(1, centre=(1, 1, 1), filled=True, color="b")

    def test_ellipsoid(self):
        plot_ellipsoid(np.diag((1, 2, 3)), color="r")  # red ellipsoid
        plot_ellipsoid(
            np.diag((1, 2, 3)), centre=(1, 2, 3), filled=True, color="y"
        )  # yellow filled ellipsoid

    def test_cylinder(self):
        plot_cylinder(radius=0.2, centre=(0.5, 0.5, 0), height=[-0.2, 0.2])
        plot_cylinder(
            radius=0.2,
            centre=(0.5, 0.5, 0),
            height=[-0.2, 0.2],
            filled=True,
            resolution=5,
            color="red",
        )

    def test_cone(self):
        plot_cone(radius=0.2, centre=(0.5, 0.5, 0), height=0.3)
        plot_cone(
            radius=0.2,
            centre=(0.5, 0.5, 0),
            height=0.3,
            filled=True,
            resolution=5,
            color="red",
        )


# ---------------------------------------------------------------------------------------#
if __name__ == "__main__":

    unittest.main(buffer=True)
