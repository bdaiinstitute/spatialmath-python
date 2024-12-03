import numpy as np
import pytest
from spatialmath.baseposelist import BasePoseList

# create a subclass to test with, its value is a list
class X(BasePoseList):
    def __init__(self, value=None, check=False):
        if value is None:
            value = [0]
        elif not isinstance(value, list):
            value = [value]
        super().__init__()
        self.data = value
        
    @staticmethod
    def _identity():
        return 0

    @property
    def shape(self):
        return (1,1)

    @staticmethod
    def isvalid(x):
        return True

class TestBasePoseList:

    def test_constructor(self):

        x = X()
        assert isinstance(x, X)
        assert len(x) == 1

    @pytest.mark.parametrize(
        'x,y,list1,expected',
        [
            (X(2), X(3), True, [6]),
            (X(2), X(3), False, 6),
        ],
    )
    def test_binop(self, x, y, list1, expected):
        assert x.binop(y, lambda x, y: x * y, list1=list1) == expected

    @pytest.mark.parametrize(
        'x,matrix,expected',
        [
            (X(2), False, [4]),
            (X(2), True, np.array(4)),
        ],
    )
    def test_unop(self, x, matrix, expected):
        result = x.unop(lambda x: 2*x, matrix=matrix)
        if isinstance(result, np.ndarray):
            assert (result == expected).all()
        else:
            assert result == expected

class TestConcreteSubclasses:
    """
    Check consistency of methods in concrete subclasses
    """
    from spatialmath import (
        SO2,
        SE2,
        SO3,
        SE3,
        Quaternion,
        UnitQuaternion,
        Twist2,
        Twist3,
        SpatialVelocity,
        SpatialAcceleration,
        SpatialForce,
        SpatialMomentum,
        Line3,
    )
    concrete_subclasses = [
        SO2,
        SE2,
        SO3,
        SE3,
        Quaternion,
        UnitQuaternion,
        Twist2,
        Twist3,
        SpatialVelocity,
        SpatialAcceleration,
        SpatialForce,
        SpatialMomentum,
        Line3,
    ]

    @pytest.mark.parametrize(
        'cls',
        concrete_subclasses,
    )
    def test_bare_init(self, cls):
        with pytest.raises(TypeError):
            cls()
