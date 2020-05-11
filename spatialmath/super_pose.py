# Created by: Aditya Dua, 2017
# Peter Corke, 2020
# 13 June, 2017

import numpy as np
from abc import ABC, abstractmethod
from collections import UserList
import copy
from spatialmath.base import argcheck 
import spatialmath.base as tr

_eps = np.finfo(np.float64).eps

# colored printing of matrices to the terminal
#   colored package has much finer control than colorama, but the latter is available by default with anaconda
try:
    from colored import fg, bg, attr
    _color = True
    #print('using colored output')
except ImportError:
    _color = False
    fg = lambda : ''
    bg = lambda : ''
    attr = lambda : ''
    
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
        super().__init__()   # enable UserList superpowers
        
    def pose_arghandler(self, arg, check=True):
        
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
                check = type(self).isvalid  # lambda function
                assert all( map( lambda x: x.shape==s and check(x), arg) ), 'all elements of list must have valid shape and value for the class'
                self.data = arg         
            elif type(arg[0]) == type(self):
                # possibly a list of objects of same type
                assert all( map( lambda x: type(x) == type(self), arg) ), 'all elements of list must have same type'
                self.data = [x.A for x in arg]
            else:
                raise ValueError('1 bad argument to constructor')
        elif type(self) == type(arg):
            # it's an object of same type, do copy
            self.data = arg.data.copy()
        else:
            raise ValueError('2 bad argument to constructor')
    
    @classmethod
    def Empty(cls):
        X = cls()
        X.data = []
        return X
    
    def append(self, x):
        #print('in append method')
        if not type(self) == type(x):
            raise ValueError("cant append different type of pose object")
        if len(x) > 1:
            raise ValueError("cant append a pose sequence - use extend")
        super().append(x.A)
        
    @property
    def A(self):
        # get the underlying numpy array
        if len(self.data) == 1:
            return self.data[0]
        else:
            return self.data

    def __getitem__(self, i):
        #print('getitem', i, 'class', self.__class__)
        #return self.__class__(self.data[i])
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
        if   type(self).__name__ == 'SO2':
            return (2,2)
        elif type(self).__name__ == 'SO3':
            return (3,3)
        elif type(self).__name__ == 'SE2':
            return (3,3)
        elif type(self).__name__ == 'SE3':
            return (4,4)
    
    def about(self):
        """
        Display succinct details of object
        
        :param self: object to display
        :type self: SO2, SE2, SO3, SE3

        Displays the type and the number of elements in compact form, eg. ``SE3[20]``.
        """
        print("{:s}[{:d}]".format( type(self).__name__, len(self)))



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
            return left.__class__(left._op2(right, lambda x, y: x @ y ))
        
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
            else:
                raise ValueError('bad operands')
        elif isinstance(right, (int, float)):
            return left._op2(right, lambda x, y: x * y )
        else:
            raise ValueError('bad operands')
        
    def __rmul__(right, left):
        """

        """
        return right.__mul__(left)
        
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
            return left.__class__(left._op2(right.inv, lambda x, y: x @ y ))
        elif isinstance(right, (int, float)):
            return left._op2(right, lambda x, y: x / y )
        else:
            raise ValueError('bad operands')

    
    def __itruediv__(left, right):
        return left.__truediv__(right)            
    

    def __add__(left, right):
        # results is not in the group, return an array, not a class
        return left._op2(right, lambda x, y: x + y )

    # def __radd__(left, right):
    #     return left.__add__(right)
    
    def __iadd__(left, right):
        return left.__add__(right)
    
    def __sub__(left, right):
        # results is not in the group, return an array, not a class
        # TODO allow class +/- a conformant array
        return left._op2(right, lambda x, y: x - y )

    # def __rsub__(left, right):
    #     return -left.__sub__(right)
    

    def __isub__(left, right):
        return left.__sub__(right)

    def __eq__(left, right):
        assert type(left) == type(right), 'operands to == are of different types'
        return left._op2(right, lambda x, y: np.allclose(x, y) )
    
    def __ne__(left, right):
        return [not x for x in self == right]
    
    def _op2(left, right, op):
        
        if isinstance(right, left.__class__):
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
                    return [op(x, y) for (x,y) in zip(left.A, right.A)]
                else:
                    raise ValueError('length of lists to == must be same length')
        elif isinstance(right, (float, int)) or (isinstance(right, np.ndarray) and right.shape == left.shape):
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
    
    def interp(self, arg):
        pass
                
    #----------------------- i/o stuff
    
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
        
    def __repr__(self):
        # #print('in __repr__')
        # if len(self) >= 1:
        #     str = ''
        #     for each in self.data:
        #         str += np.array2string(each) + '\n\n'
        #     return str.rstrip("\n")  # Remove trailing newline character
        # else:
        #      raise ValueError('no elements in the value list')
        return self.__str__()

    def __str__(self):
        #print('in __str__')
        def mformat(self, X):
            # X is an ndarray value to be display
            # self provides set type for formatting
            out = ''
            n = self.N  # dimension of rotation submatrix
            for rownum, row in enumerate(X):
                rowstr = '  '
                # format the columns
                for colnum, element in enumerate(row):
                    if abs(element) < 10 * _eps:
                        element = 0
                    s = '{:< 10g}'.format(element)

                    if rownum < n:
                        if colnum < n:
                            # rotation part
                            s = fg('red') + bg('grey_93') + s + attr(0)
                        else:
                            # translation part
                            s = fg('blue') + bg('grey_93') + s + attr(0)
                    else:
                        # bottom row
                        s = fg('grey_50') + bg('grey_93') + s + attr(0)
                    rowstr += s
                out += rowstr + bg('grey_93') + '  ' + attr(0) + '\n'
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
