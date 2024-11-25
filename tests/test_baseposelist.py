import numpy as np
from spatialmath.baseposelist import BasePoseList

# create a subclass to test with, its value is a scalar
class X(BasePoseList):
    def __init__(self, value=0, check=False):
        super().__init__()
        self.data = [value]
        
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

        x = X.Empty()
        assert isinstance(x, X)
        assert len(x) == 0

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

    def test_binop(self):
        x = X(2)
        y = X(3)

        # singelton x singleton
        assert x.binop(y, lambda x, y: x * y) == [6]
        assert x.binop(y, lambda x, y: x * y, list1=False) == 6

        y = X.Alloc(5)
        for i in range(0, 5):
            y[i] = X(i + 1)

        # singelton x non-singleton
        assert x.binop(y, lambda x, y: x * y) == [2, 4, 6, 8, 10]
        assert x.binop(y, lambda x, y: x * y, list1=False) == [2, 4, 6, 8, 10]

        # non-singelton x singleton
        assert y.binop(x, lambda x, y: x * y) == [2, 4, 6, 8, 10]
        assert y.binop(x, lambda x, y: x * y, list1=False) == [2, 4, 6, 8, 10]

        # non-singelton x non-singleton
        assert y.binop(y, lambda x, y: x * y) == [1, 4, 9, 16, 25]
        assert y.binop(y, lambda x, y: x * y, list1=False) == [1, 4, 9, 16, 25]

    def test_unop(self):
        x = X(2)

        f = lambda x: 2 * x

        assert x.unop(f) == [4]
        assert x.unop(f, matrix=True) == np.r_[4]

        x = X.Alloc(5)
        for i in range(0, 5):
            x[i] = X(i + 1)

        assert x.unop(f) == [2, 4, 6, 8, 10]
        y = x.unop(f, matrix=True)
        assert y.shape == (5,1)
        assert np.all(y - np.c_[2, 4, 6, 8, 10].T == 0)

    def test_arghandler(self):
        pass
