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

    def test_empty(self):
        x = X.Empty()
        assert isinstance(x, X)
        assert len(x) == 0

    def test_alloc(self):
        x = X.Alloc(10)
        assert isinstance(x, X)
        assert len(x) == 10
        for xx in x:
            assert xx.A == 0

    def test_setget(self):
        x = X.Alloc(10)
        for i in range(0, 10):
            x[i] = X(2 * i)

        for i,v in enumerate(x):
            assert v.A == 2 * i

    def test_append(self):
        x = X.Empty()
        for i in range(0, 10):
            x.append(X(i+1))
        assert len(x) == 10
        assert [xx.A for xx in x] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    def test_extend(self):
        x = X.Alloc(5)
        for i in range(0, 5):
            x[i] = X(i + 1)
        y = X.Alloc(5)
        for i in range(0, 5):
            y[i] = X(i + 10)
        x.extend(y)
        assert len(x) == 10
        assert [xx.A for xx in x] == [1, 2, 3, 4, 5, 10, 11, 12, 13, 14]
        
    def test_insert(self):
        x = X.Alloc(10)
        for i in range(0, 10):
            x[i] = X(i + 1)
        x.insert(5, X(100))
        assert len(x) == 11
        assert [xx.A for xx in x] == [1, 2, 3, 4, 5, 100, 6, 7, 8, 9, 10]

    def test_pop(self):
        x = X.Alloc(10)
        for i in range(0, 10):
            x[i] = X(i + 1)

        y = x.pop()
        assert len(y) == 1
        assert y.A == 10
        assert len(x) == 9
        assert [xx.A for xx in x] == [1, 2, 3, 4, 5, 6, 7, 8, 9]

    def test_clear(self):
        x = X.Alloc(10)
        x.clear()
        assert len(x) == 0

    def test_reverse(self):
        x = X.Alloc(5)
        for i in range(0, 5):
            x[i] = X(i + 1)
        x.reverse()
        assert len(x) == 5
        assert [xx.A for xx in x] == [5, 4, 3, 2, 1]

    @pytest.mark.parametrize(
        'x,y,list1,expected',
        [
            (X(2), X(3), True, [6]),
            (X(2), X(3), False, 6),
            (X(2), X([1,2,3,4,5]), True, [2,4,6,8,10]),
            (X(2), X([1,2,3,4,5]), False, [2,4,6,8,10]),
            (X([1,2,3,4,5]), X(2), True, [2,4,6,8,10]),
            (X([1,2,3,4,5]), X(2), False, [2,4,6,8,10]),
            (X([1,2,3,4,5]), X([1,2,3,4,5]), True, [1,4,9,16,25]),
            (X([1,2,3,4,5]), X([1,2,3,4,5]), False, [1,4,9,16,25]),
        ],
    )
    def test_binop(self, x, y, list1, expected):
        assert x.binop(y, lambda x, y: x * y, list1=list1) == expected

    @pytest.mark.parametrize(
        'x,matrix,expected',
        [
            (X(2), False, [4]),
            (X(2), True, np.array(4)),
            (X([1,2,3,4,5]), False, [2,4,6,8,10]),
            (X([1,2,3,4,5]), True, np.array([[2,4,6,8,10]]).T),
        ],
    )
    def test_unop(self, x, matrix, expected):
        result = x.unop(lambda x: 2*x, matrix=matrix)
        if isinstance(result, np.ndarray):
            assert (result == expected).all()
        else:
            assert result == expected
