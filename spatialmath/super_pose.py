# Created by: Aditya Dua, 2017
# Peter Corke, 2020
# 13 June, 2017

import numpy as np
import sympy
from abc import ABC, abstractmethod
from collections import UserList
import copy
from spatialmath.base import argcheck
import spatialmath.base as tr
#from spatialmath import Plucker


_eps = np.finfo(np.float64).eps

# colored printing of matrices to the terminal
#   colored package has much finer control than colorama, but the latter is available by default with anaconda
try:
    from colored import fg, bg, attr
    _color = True
    #print('using colored output')
except ImportError:
    #print('colored not found')
    _color = False


# try:
#     import colorama
#     colorama.init()
#     print('using colored output')
#     from colorama import Fore, Back, Style

# except:
#     class color:
#         def __init__(self):
#             self.RED = ''
#             self.BLUE = ''
#             self.BLACK = ''
#             self.DIM = ''

# print(Fore.RED + '1.00 2.00 ' + Fore.BLUE + '3.00')
# print(Fore.RED + '1.00 2.00 ' + Fore.BLUE + '3.00')
# print(Fore.BLACK + Style.DIM + '0 0 1')


class SMPose(UserList, ABC):
    # inherits from:
    #  UserList, gives list-like functionality
    #  ABC, defines an abstract class, can't be instantiated

    #    @property
    #    def length(self):
    #        """
    #        Property to return number of matrices in pose object
    #        :return: int
    #        """
    #        return len(self._list)
    #
    #    @property
    #    def data(self):
    #        """
    #        Always returns a list containing the matrices of the pose object.
    #        :return: A list of matrices.
    #        """
    #        return self._list
    #
    #
    #    def is_equal(self, other):
    #        if (type(self) is type(other)) and (self.length == other.length):
    #            for i in range(self.length):
    #                try:
    #                    npt.assert_almost_equal(self.data[i], other.data[i])
    #                except AssertionError:
    #                    return False
    #            return True
    #
    #    def append(self, item):
    #        check_args.super_pose_appenditem(self, item)
    #        if type(item) is np.matrix:
    #            self._list.append(item)
    #        else:
    #            for each_matrix in item:
    #                self._list.append(each_matrix)
    #
    #    def tr_2_rt(self):
    #        assert isinstance(self, pose.SE2) or isinstance(self, pose.SE3)
    #
    #    def t_2_r(self):
    #        assert isinstance(self, pose.SE2) or isinstance(self, pose.SE3)
    #        for each_matrix in self:
    #            pass  # TODO

    def __init__(self):
        # handle common cases
        #  deep copy
        #  numpy array
        #  list of numpy array
        # validity checking??
        # TODO should this be done by __new__?
        super().__init__()   # enable UserList superpowers

    def pose_arghandler(self, arg, check=True):
        """
        Generalized argument handling for pose classes
        
        :param arg: value of pose
        :param check: check type of argument, defaults to True
        :type check: TYPE, optional
        :raises ValueError: bad type passed

        The argument can be any of:
            
        1. a numpy.ndarray of the appropriate shape and valid value for the subclass
        2. an instance of the subclass
        3. a list whose elements all meet the criteria of 1.
        4. a list whose elements are all instances of the subclass
        
        Examples::
            
            SE3( np.identity(4))
            SE3( SE3() )
            SE3( [np.identity(4), np.identity(4)])
            SE3( [SE3(), SE3()])

        """

        if isinstance(arg, np.ndarray):
            # it's a numpy array
            assert arg.shape == self.shape, 'array must have valid shape for the class'
            assert type(self).isvalid(arg), 'array must have valid value for the class'
            self.data.append(arg)
        elif isinstance(arg, list):
            # construct from a list

            if isinstance(arg[0], np.ndarray):
                #print('list of numpys')
                # possibly a list of numpy arrays
                s = self.shape
                if check:
                    checkfunc = type(self).isvalid # lambda function
                    assert all(map(lambda x: x.shape == s and checkfunc(x), arg)), 'all elements of list must have valid shape and value for the class'
                else:
                    assert all(map(lambda x: x.shape == s, arg))
                self.data = arg
            elif type(arg[0]) == type(self):
                # possibly a list of objects of same type
                assert all(map(lambda x: type(x) == type(self), arg)), 'all elements of list must have same type'
                self.data = [x.A for x in arg]
            else:
                raise ValueError('bad list argument to constructor')
        elif type(self) == type(arg):
            # it's an object of same type, do copy
            self.data = arg.data.copy()
        else:
            raise ValueError('bad argument to constructor')

    @classmethod
    def Empty(cls):
        """
        Construct an empy pose object
        
        :param cls: The pose subclass
        :type cls: type
        :return: a pose object with zero lenght
        :rtype: subclass instance

        Example::
            
            >>> x = SE3()
            >>> len(x)
            1
            >>> x = SE3.Empty()
            >>> len(x)
            1
            
        """
        X = cls()
        X.data = []
        return X

    def append(self, x):
        """
        Append a pose object
        
        :param x: A pose subclass
        :type x: subclass
        :raises ValueError: incorrect type of appended object

        Appends the argument to the object's internal list.
        
        Examples::
            
            >>> x = SE3()
            >>> len(x)
            1
            >>> x.append(SE3())
            >>> len(x)
            2
        """
        #print('in append method')
        if not type(self) == type(x):
            raise ValueError("cant append different type of pose object")
        if len(x) > 1:
            raise ValueError("cant append a pose sequence - use extend")
        super().append(x.A)

    @property
    def A(self):
        """
        Access the underlying array
        
        :return: The numeric array
        :rtype: numpy.ndarray
        
        Each pose subclass is stored internally as a numpy array. This property returns
        the array, shape depends on the particular subclass.
        
        Examples::
            
        >>> x = SE3()
        >>> x.A
        array([[1., 0., 0., 0.],
               [0., 1., 0., 0.],
               [0., 0., 1., 0.],
               [0., 0., 0., 1.]])
        """
        # get the underlying numpy array
        if len(self.data) == 1:
            return self.data[0]
        else:
            return self.data

    def __getitem__(self, i):
        """
        Access
        :param i: index into the internal list
        :type i: int
        :return: one element of internal list
        :rtype: subclass
        :raises IndexError: if the element is out of bounds

        Note that only a single index is supported, slices are not.
        
        Example::
            
        >>> x = SE3.Rx([0, math.pi/2, math.pi])
        >>> len(x)
        3
        >>> x[1]
           1           0           0           0            
           0           0          -1           0            
           0           1           0           0            
           0           0           0           1  

        """
        # print('getitem', i, 'class', self.__class__)
        # return self.__class__(self.data[i])
        if isinstance(i, slice):
            return self.__class__([self.data[k] for k in range(i.start or 0, i.stop or len(self), i.step or 1)])
        else:
            return self.__class__(self.data[i])

    #----------------------- tests
    @property
    def isSO(self):
        """
        Test if object belongs to SO(n)

        :param self: object to test
        :return: true if object is instance of SO2 or SO3
        :rtype: bool
        """
        return type(self).__name__ == 'SO2' or type(self).__name__ == 'SO3'

    @property
    def isSE(self):
        """
        Test if object belongs to SE(n)

        :param self: object to test
        :return: true if object is instance of SE2 or SE3
        :rtype: bool
        """
        return type(self).__name__ == 'SE2' or type(self).__name__ == 'SE3'

    @property
    def N(self):
        """
        Dimension of the object's space

        :param self: object to test
        :return: 2 for SO2 or SE2, 3 for SO3 or SE3
        :rtype: int
        """
        if type(self).__name__ == 'SO2' or type(self).__name__ == 'SE2':
            return 2
        else:
            return 3

    # compatibility methods

    def isrot(self):
        """
        Test if object belongs to SO(3)

        :param self: object to test
        :return: true if object is instance of SO3
        :rtype: bool
        """
        return type(self).__name__ == 'SO3'

    def isrot2(self):
        """
        Test if object belongs to SO(2)

        :param self: object to test
        :return: true if object is instance of SO2
        :rtype: bool
        """
        return type(self).__name__ == 'SO2'

    def ishom(self):
        """
        Test if object belongs to SE(3)

        :param self: object to test
        :return: true if object is instance of SE3
        :rtype: bool
        """
        return type(self).__name__ == 'SE3'

    def ishom2(self):
        """
        Test if object belongs to SE(2)

        :param self: object to test
        :return: true if object is instance of SE2
        :rtype: bool
        """
        return type(self).__name__ == 'SE2'

    #----------------------- properties
    @property
    def shape(self):
        """
        Dimension of the object's underlying matrix representation

        :param self: object to test
        :return: (2,2) for SO2, (3,3) for SE2 and SO3, and (4,4) for SE3
        :rtype: int
        """
        if type(self).__name__ == 'SO2':
            return (2, 2)
        elif type(self).__name__ == 'SO3':
            return (3, 3)
        elif type(self).__name__ == 'SE2':
            return (3, 3)
        elif type(self).__name__ == 'SE3':
            return (4, 4)

    def about(self):
        """
        Display succinct details of object

        :param self: object to display
        :type self: SO2, SE2, SO3, SE3

        Displays the type and the number of elements in compact form, eg. ``SE3[20]``.
        """
        print("{:s}[{:d}]".format(type(self).__name__, len(self)))

    #----------------------- arithmetic

    def __mul__(left, right):
        """
        Pose multiplication

        :arg left: left multiplicand
        :arg right: right multiplicand
        :return: product
        :raises: ValueError

        - ``X * Y`` compounds the poses X and Y
        - ``X * s`` performs elementwise multiplication of the elements of ``X``
        - ``s * X`` performs elementwise multiplication of the elements of ``X``
        - ``X * v`` transforms the vector.

        ==============   ==============   ===========  ===================
                   Multiplicands                   Product
        -------------------------------   --------------------------------
            left             right            type           result
        ==============   ==============   ===========  ===================
        Pose             Pose             Pose         matrix product
        Pose             scalar           matrix       elementwise product
        scalar           Pose             matrix       elementwise product
        Pose             N-vector         N-vector     vector transform
        Pose             NxM matrix       NxM matrix   vector transform
        ==============   ==============   ===========  ===================

        Any other input combinations result in a ValueError.

        Note that left and right can have a length greater than 1 in which case

        ====   =====   ====  ================================
        left   right   len     operation
        ====   =====   ====  ================================
         1      1       1    ``prod = left * right``
         1      M       M    ``prod[i] = left * right[i]``
         N      1       M    ``prod[i] = left[i] * right``
         M      M       M    ``prod[i] = left[i] * right[i]``
        ====   =====   ====  ================================

        A scalar of length M is list, tuple or numpy array.
        An N-vector of length M is a NxM numpy array, where each column is an N-vector.
        """
        if isinstance(left, right.__class__):
            #print('*: pose x pose')
            return left.__class__(left._op2(right, lambda x, y: x @ y))

        elif isinstance(right, (list, tuple, np.ndarray)):
            #print('*: pose x array')
            if len(left) == 1 and argcheck.isvector(right, left.N):
                # pose x vector
                #print('*: pose x vector')
                v = argcheck.getvector(right, out='col')
                if left.isSE:
                    # SE(n) x vector
                    return tr.h2e(left.A @ tr.e2h(v))
                else:
                    # SO(n) x vector
                    return left.A @ v

            elif len(left) > 1 and argcheck.isvector(right, left.N):
                # pose array x vector
                #print('*: pose array x vector')
                v = argcheck.getvector(right)
                if left.isSE:
                    # SE(n) x vector
                    v = tr.e2h(v)
                    return np.array([tr.h2e(x @ v).flatten() for x in left.A]).T
                else:
                    # SO(n) x vector
                    return np.array([(x @ v).flatten() for x in left.A]).T

            elif len(left) == 1 and isinstance(right, np.ndarray) and left.isSO and right.shape[0] == left.N:
                # SO(n) x matrix
                return left.A @ right
            elif len(left) == 1 and isinstance(right, np.ndarray) and left.isSE and right.shape[0] == left.N:
                # SE(n) x matrix
                return tr.h2e(left.A @ tr.e2h(right))
            elif isinstance(right, np.ndarray) and left.isSO and right.shape[0] == left.N and len(left) == right.shape[1]:
                # SO(n) x matrix
                return np.c_[[x.A @ y for x,y in zip(right, left.T)]].T
            elif isinstance(right, np.ndarray) and left.isSE and right.shape[0] == left.N and len(left) == right.shape[1]:
                # SE(n) x matrix
                return np.c_[[tr.h2e(x.A @ tr.e2h(y)) for x,y in zip(right, left.T)]].T
            else:
                raise ValueError('bad operands')
        elif isinstance(right, (int, np.int64, float, np.float64)):
            return left._op2(right, lambda x, y: x * y)
        else:
            return NotImplemented
        
    def __rmul__(right, left):
        """
        """
        if isinstance(left, (int, np.int64, float, np.float64)):
            return right.__mul__(left)
        else:
            return NotImplemented

    def __imul__(left, right):
        """
        Pose multiplication

        :arg left: left multiplicand
        :arg right: right multiplicand
        :return: product
        :raises: ValueError

        - ``X *= Y`` compounds the poses X and Y and places the result in X
        - ``X *= s`` performs elementwise multiplication of the elements of ``X``


        Any other input combinations result in a ValueError.

        Note that left and right can have a length greater than 1 in which case

        ====   =====   ====  ================================
        left   right   len     operation
        ====   =====   ====  ================================
         1      1       1    ``prod = left * right``
         1      M       M    ``prod[i] = left * right[i]``
         N      1       M    ``prod[i] = left[i] * right``
         M      M       M    ``prod[i] = left[i] * right[i]``
        ====   =====   ====  ================================

        A scalar of length M is list, tuple or numpy array.
        An N-vector of length M is a NxM numpy array, where each column is an N-vector.
        """
        return left.__mul__(right)

    def __pow__(self, n):
        assert type(n) is int, 'exponent must be an int'
        return self.__class__([np.linalg.matrix_power(x, n) for x in self.data])

    def __ipow__(self, n):
        return self.__pow__(n)

    def __truediv__(left, right):
        if isinstance(left, right.__class__):
            return left.__class__(left._op2(right.inv, lambda x, y: x @ y))
        elif isinstance(right, (int, np.int64, float, np.float64)):
            return left._op2(right, lambda x, y: x / y)
        else:
            raise ValueError('bad operands')

    def __itruediv__(left, right):
        return left.__truediv__(right)

    def __add__(left, right):
        # results is not in the group, return an array, not a class
        return left._op2(right, lambda x, y: x + y)

    def __radd__(left, right):
        return left.__add__(right)

    def __iadd__(left, right):
        return left.__add__(right)

    def __sub__(left, right):
        # results is not in the group, return an array, not a class
        # TODO allow class +/- a conformant array
        return left._op2(right, lambda x, y: x - y)

    def __rsub__(left, right):
        return -left.__sub__(right)

    def __isub__(left, right):
        return left.__sub__(right)

    def __eq__(left, right):
        assert type(left) == type(right), 'operands to == are of different types'
        return left._op2(right, lambda x, y: np.allclose(x, y))

    def __ne__(left, right):
        return [not x for x in self == right]

    def _op2(left, right, op):

        if isinstance(right, left.__class__):
            # class by class
            if len(left) == 1:
                if len(right) == 1:
                    #print('== 1x1')
                    return op(left.A, right.A)
                else:
                    #print('== 1xN')
                    return [op(left.A, x) for x in right.A]
            else:
                if len(right) == 1:
                    #print('== Nx1')
                    return [op(x, right.A) for x in left.A]
                elif len(left) == len(right):
                    #print('== NxN')
                    return [op(x, y) for (x, y) in zip(left.A, right.A)]
                else:
                    raise ValueError('length of lists to == must be same length')
        elif isinstance(right, (float, int)) or (isinstance(right, np.ndarray) and right.shape == left.shape):
            # class by matrix
            if len(left) == 1:
                return op(left.A, right)
            else:
                return [op(x, right) for x in left.A]

    # @classmethod
    # def rand(cls):
    #     obj = cls(uniform(0, 360), unit='deg')
    #     return obj

     #----------------------- functions

    def exp(self, arg):
        pass

    def log(self, arg):
        pass

    def interp(self, T1=None, s=None):
        if self.N == 2:
            return self.__class__(tr.trinterp2(self.A, T1.A, s))
        elif self.N == 3:
            return self.__class__(tr.trinterp(self.A, T1.A, s))

    # ----------------------- i/o stuff

    def print(self):
        print(self)

    def printline(self):
        if self.N == 2:
            tr.trprint2(self.A)
        else:
            tr.trprint(self.A)

    def plot(self, *args, **kwargs):
        if self.N == 2:
            tr.trplot2(self.A, *args, **kwargs)
        else:
            tr.trplot(self.A, *args, **kwargs)
            
    def animate(self, *args, T0=None, **kwargs):
        if T0 is not None:
            T0 = T0.A
        if self.N == 2:
            tr.tranimate2(self.A, T0=T0, *args, **kwargs)
        else:
            tr.tranimate(self.A, T0=T0, *args, **kwargs)

    def __repr__(self):
        # #print('in __repr__')
        # if len(self) >= 1:
        #     str = ''
        #     for each in self.data:
        #         str += np.array2string(each) + '\n\n'
        #     return str.rstrip("\n")  # Remove trailing newline character
        # else:
        #      raise ValueError('no elements in the value list')
        return self._string(color=_color)

    def __str__(self):
        return self._string(color=False)

    def _string(self, color=False, squash=True):
        #print('in __str__')

        FG = lambda c: fg(c) if color else ''
        BG = lambda c: bg(c) if color else ''
        ATTR = lambda c: attr(c) if color else ''

        def mformat(self, X):
            # X is an ndarray value to be display
            # self provides set type for formatting
            out = ''
            n = self.N  # dimension of rotation submatrix
            for rownum, row in enumerate(X):
                rowstr = '  '
                # format the columns
                for colnum, element in enumerate(row):
                    if isinstance(element, sympy.Expr):
                        s = '{:<12s}'.format(str(element))
                    else:
                        if squash and abs(element) < 10 * _eps:
                            element = 0
                        s = '{:< 12g}'.format(element)

                    if rownum < n:
                        if colnum < n:
                            # rotation part
                            s = FG('red') + BG('grey_93') + s + ATTR(0)
                        else:
                            # translation part
                            s = FG('blue') + BG('grey_93') + s + ATTR(0)
                    else:
                        # bottom row
                        s = FG('grey_50') + BG('grey_93') + s + ATTR(0)
                    rowstr += s
                out += rowstr + BG('grey_93') + '  ' + ATTR(0) + '\n'
            return out

        output_str = ''

        if len(self.data) == 0:
            output_str = '[]'
        elif len(self.data) == 1:
            # single matrix case
            output_str = mformat(self, self.A)
        else:
            # sequence case
            for count, X in enumerate(self.data):
                # add separator lines and the index
                output_str += fg('green') + '[{:d}] =\n'.format(count) + attr(0) + mformat(self, X)

        return output_str
    
    
class SMTwist(UserList, ABC):

    # ------------------------- list support -------------------------------#
    def __init__(self):
        # handle common cases
        #  deep copy
        #  numpy array
        #  list of numpy array
        # validity checking??
        # TODO should this be done by __new__?
        super().__init__()   # enable UserList superpowers
        
    @classmethod
    def Empty(cls):
        """
        Construct an empy pose object
        
        :param cls: The pose subclass
        :type cls: type
        :return: a pose object with zero lenght
        :rtype: subclass instance

        Example::
            
            >>> x = SE3()
            >>> len(x)
            1
            >>> x = SE3.Empty()
            >>> len(x)
            1
            
        """
        X = cls()
        X.data = []
        return X
        
    def __getitem__(self, i):
        # print('getitem', i, 'class', self.__class__)
        if isinstance(i, slice):
            return self.__class__([self.data[k] for k in range(i.start or 0, i.stop or len(self), i.step or 1)], check=False)
        else:
            return self.__class__(self.data[i], check=False)
    
    def append(self, x):
        """
        Append a pose object
        
        :param x: A pose subclass
        :type x: subclass
        :raises ValueError: incorrect type of appended object

        Appends the argument to the object's internal list.
        
        Examples::
            
            >>> x = SE3()
            >>> len(x)
            1
            >>> x.append(SE3())
            >>> len(x)
            2
        """
        #print('in append method')
        if not type(self) == type(x):
            raise ValueError("cant append different type of pose object")
        if len(x) > 1:
            raise ValueError("cant append a pose sequence - use extend")
        super().append(x.S)
        
    @property
    def S(self):
        """
        Twist vector

        TW.S is the twist vector in se(3) as a vector (6x1).

        Notes:

        - Sometimes referred to as the twist coordinate vector.
        """
        # get the underlying numpy array
        if len(self.data) == 1:
            return self.data[0]
        else:
            return self.data
    
    @property
    def v(self):
        return self.data[0][:3]
    
    @property
    def w(self):
        return self.data[0][3:6]

    @property
    def isprismatic(self):
        return tr.iszerovec(self.w)
    

    def prod(self):
        """
        %Twist.prod Compound array of twists
        %
        TW.prod is a twist representing the product (composition) of the
        successive elements of TW (1xN), an array of Twists.
                %
                %
        See also RTBPose.prod, Twist.mtimes.
        """
        out = self[0]
        
        for t in self[1:]:
            out *= t
        return out

