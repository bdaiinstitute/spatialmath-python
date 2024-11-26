import pytest
import unittest
import math

try:
    import sympy as sp

    _symbolics = True
except ImportError:
    _symbolics = False

from spatialmath.base.symbolic import *


class Test_symbolic(unittest.TestCase):
    @pytest.mark.skipif(not _symbolics, reason="sympy required")
    def test_symbol(self):
        theta = symbol("theta")
        assert isinstance(theta, sp.Expr)
        assert theta.is_real

        theta = symbol("theta", real=False)
        assert isinstance(theta, sp.Expr)
        assert not theta.is_real

        theta, psi = symbol("theta, psi")
        assert isinstance(theta, sp.Expr)
        assert isinstance(psi, sp.Expr)

        theta, psi = symbol("theta psi")
        assert isinstance(theta, sp.Expr)
        assert isinstance(psi, sp.Expr)

        q = symbol("q:6")
        assert len(q) == 6
        for _ in q:
            assert isinstance(_, sp.Expr)
            assert _.is_real

    @pytest.mark.skipif(not _symbolics, reason="sympy required")
    def test_issymbol(self):
        theta = symbol("theta")
        assert not issymbol(3)
        assert not issymbol("not a symbol")
        assert not issymbol([1, 2])
        assert issymbol(theta)

    @pytest.mark.skipif(not _symbolics, reason="sympy required")
    def test_functions(self):

        theta = symbol("theta")
        assert isinstance(sin(theta), sp.Expr)
        assert isinstance(sin(1.0), float)

        assert isinstance(cos(theta), sp.Expr)
        assert isinstance(cos(1.0), float)

        assert isinstance(sqrt(theta), sp.Expr)
        assert isinstance(sqrt(1.0), float)

        x = (theta - 1) * (theta + 1) - theta ** 2
        assert math.isclose(simplify(x).evalf(), -1)

    @pytest.mark.skipif(not _symbolics, reason="sympy required")
    def test_constants(self):

        x = zero()
        assert isinstance(x, sp.Expr)
        assert math.isclose(x.evalf(), 0)

        x = one()
        assert isinstance(x, sp.Expr)
        assert math.isclose(x.evalf(), 1)

        x = negative_one()
        assert isinstance(x, sp.Expr)
        assert math.isclose(x.evalf(), -1)

        x = pi()
        assert isinstance(x, sp.Expr)
        assert math.isclose(x.evalf(), math.pi)


# ---------------------------------------------------------------------------------------#
if __name__ == "__main__":  # pragma: no cover

    unittest.main()
