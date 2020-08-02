# Created by: Aditya Dua, 2017
# Peter Corke, 2020
# 13 June, 2017

import numpy as np
import sympy
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
    print('using colored output')
except:
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


class SMPose(UserList):
    """
    Superclass for SO(N) and SE(N) objects

    Subclasses are:

    - ``SO2`` representing elements of SO(2) which describe rotations in 2D
    - ``SE2`` representing elements of SE(2) which describe rigid-body motion in 2D
    - ``SO3`` representing elements of SO(3) which describe rotations in 3D
    - ``SE3`` representing elements of SE(3) which describe rigid-body motion in 3D

    Arithmetic operators are overloaded but the operation they perform depend
    on the types of the operands.  For example:

    - ``*`` will compose two instances of the same subclass, and the result will be
      an instance of the same subclass, since this is a group operator.
    - ``+`` will add two instances of the same subclass, and the result will be
      a matrix, not an instance of the same subclass, since addition is not a group operator.

    These classes all inherit from ``UserList`` which enables them to 
    represent a sequence of values, ie. an ``SE3`` instance can contain
    a sequence of SE(3) values.  Most of the Python ``list`` operators
    are applicable::

        >>> x = SE3()  # new instance with identity matrix value
        >>> len(x)     # it is a sequence of one value
        1
        >>> x.append(x)  # append to itself
        >>> len(x)       # it is a sequence of two values
        2
        >>> x[1]         # the element has a 4x4 matrix value
        SE3([
        array([[1., 0., 0., 0.],
               [0., 1., 0., 0.],
               [0., 0., 1., 0.],
            [0., 0., 0., 1.]]) ])
        >>> x[1] = SE3.Rx(0.3)  # set an elements of the sequence
        >>> x.reverse()         # reverse the elements in the sequence
        >>> del x[1]            # delete an element

    """

    def __new__(cls, *args, **kwargs):
        """
        Create the subclass instance (superclass method)

        Create a new instance and call the superclass initializer to enable the 
        ``UserList`` capabilities.
        """

        pose = super(SMPose, cls).__new__(cls)  # create a new instance
        super().__init__(pose)  # initialize UserList
        return pose

    def _arghandler(self, arg, check=True):
        """
        Assign value to pose subclasses (superclass method)
        
        :param self: the pose object to be set
        :type self: SO2, SE2, SO3, SE3 instance
        :param arg: value of pose
        :param check: check type of argument, defaults to True
        :type check: TYPE, optional
        :raises ValueError: bad type passed

        The value ``arg`` can be any of:
            
        # a numpy.ndarray of the appropriate shape and value which is valid for the subclass
        # a list whose elements all meet the criteria above
        # an instance of the subclass
        # a list whose elements are all instances of the subclass
        
        Examples::

            SE3( np.identity(4))
            SE3( [np.identity(4), np.identity(4)])
            SE3( SE3() )
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
        Construct a new pose object with zero items (superclass method)
        
        :param cls: The pose subclass
        :type cls: SO2, SE2, SO3, SE3
        :return: a pose with zero values
        :rtype: SO2, SE2, SO3, SE3 instance

        This constructs an empty pose container which can be appended to.  For example::
            
            >>> x = SO2.Empty()
            >>> len(x)
            0
            >>> x.append(SO2(20, 'deg'))
            >>> len(x)
            1
            
        """
        X = cls()
        X.data = []
        return X

# ------------------------------------------------------------------------ #

    @property
    def A(self):
        """
        Interal array representation (superclass property)
        
        :param self: the pose object
        :type self: SO2, SE2, SO3, SE3 instance
        :return: The numeric array
        :rtype: numpy.ndarray
        
        Each pose subclass SO(N) or SE(N) are stored internally as a numpy array. This property returns
        the array, shape depends on the particular subclass.
        
        Examples::
            
            >>> x = SE3()
            >>> x.A
            array([[1., 0., 0., 0.],
                   [0., 1., 0., 0.],
                   [0., 0., 1., 0.],
                   [0., 0., 0., 1.]])

        :seealso: `shape`, `N`
        """
        # get the underlying numpy array
        if len(self.data) == 1:
            return self.data[0]
        else:
            return self.data
        
    @property
    def shape(self):
        """
        Shape of the object's matrix representation (superclass property)

        :return: matrix shape
        :rtype: 2-tuple of ints

        (2,2) for ``SO2``, (3,3) for ``SE2`` and ``SO3``, and (4,4) for ``SE3``.
        
        Example::
            
            >>> x = SE3()
            >>> x.shape
            (4, 4)
        """
        if type(self).__name__ == 'SO2':
            return (2, 2)
        elif type(self).__name__ == 'SO3':
            return (3, 3)
        elif type(self).__name__ == 'SE2':
            return (3, 3)
        elif type(self).__name__ == 'SE3':
            return (4, 4)

    @property
    def about(self):
        """
        Succinct summary of object type and length (superclass property)

        :return: succinct summary
        :rtype: str

        Displays the type and the number of elements in compact form, for 
        example::

            >>> x = SE3([SE3() for i in range(20)])
            >>> len(x)
            20
            >>> print(x.about)
            SE3[20]
        """
        return "{:s}[{:d}]".format(type(self).__name__, len(self))
    
    @property
    def N(self):
        """
        Dimension of the object's group (superclass property)

        :return: dimension
        :rtype: int

        Dimension of the group is 2 for ``SO2`` or ``SE2``, and 3 for ``SO3`` or ``SE3``.
        This corresponds to the dimension of the space, 2D or 3D, to which these
        rotations or rigid-body motions apply.
        
        Example::
            
            >>> x = SE3()
            >>> x.N
            3
        """
        if type(self).__name__ == 'SO2' or type(self).__name__ == 'SE2':
            return 2
        else:
            return 3

    #----------------------- tests
    @property
    def isSO(self):
        """
        Test if object belongs to SO(n) group (superclass property)

        :param self: object to test
        :type self: SO2, SE2, SO3, SE3 instance
        :return: ``True`` if object is instance of SO2 or SO3
        :rtype: bool
        """
        return type(self).__name__ == 'SO2' or type(self).__name__ == 'SO3'

    @property
    def isSE(self):
        """
        Test if object belongs to SE(n) group (superclass property)

        :param self: object to test
        :type self: SO2, SE2, SO3, SE3 instance
        :return: ``True`` if object is instance of SE2 or SE3
        :rtype: bool
        """
        return type(self).__name__ == 'SE2' or type(self).__name__ == 'SE3'


        
# ------------------------------------------------------------------------ #

    def __getitem__(self, i):
        """
        Access value of a pose object (superclass method)

        :param i: index of element to return
        :type i: int
        :return: the specific element of the pose
        :rtype: SO2, SE2, SO3, SE3 instance
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

        if isinstance(i, slice):
            return self.__class__([self.data[k] for k in range(i.start or 0, i.stop or len(self), i.step or 1)])
        else:
            return self.__class__(self.data[i])
        
    def __setitem__(self, i, value):
        """
        Assign a value to a pose object (superclass method)
        
        :param i: index of element to assign to
        :type i: int
        :param value: the value to insert
        :type value: SO2, SE2, SO3, SE3 instance
        :raises ValueError: incorrect type of assigned value

        Assign the argument to an element of the object's internal list of values.
        This supports the assignement operator, for example::
            
            >>> x = SE3([SE3() for i in range(10)]) # sequence of ten identity values
            >>> len(x)
            10
            >>> x[3] = SE3.Rx(0.2)   # assign to position 3 in the list
        """
        if not type(self) == type(value):
            raise ValueError("cant append different type of pose object")
        if len(value) > 1:
            raise ValueError("cant insert a pose sequence - must have len() == 1")
        self.data[i] = value.A

    def append(self, x):
        """
        Append a value to a pose object (superclass method)
        
        :param x: the value to append
        :type x: SO2, SE2, SO3, SE3 instance
        :raises ValueError: incorrect type of appended object

        Appends the argument to the object's internal list of values.
        
        Examples::
            
            >>> x = SE3()
            >>> len(x)
            1
            >>> x.append(SE3.Rx(0.1))
            >>> len(x)
            2
        """
        #print('in append method')
        if not type(self) == type(x):
            raise ValueError("cant append different type of pose object")
        if len(x) > 1:
            raise ValueError("cant append a pose sequence - use extend")
        super().append(x.A)
        

    def extend(self, x):
        """
        Extend sequence of values of a pose object (superclass method)
        
        :param x: the value to extend
        :type x: SO2, SE2, SO3, SE3 instance
        :raises ValueError: incorrect type of appended object

        Appends the argument to the object's internal list of values.
        
        Examples::
            
            >>> x = SE3()
            >>> len(x)
            1
            >>> x.append(SE3.Rx(0.1))
            >>> len(x)
            2
        """
        #print('in extend method')
        if not type(self) == type(x):
            raise ValueError("cant append different type of pose object")
        if len(x) == 0:
            raise ValueError("cant extend a singleton pose  - use append")
        super().extend(x.A)

    def insert(self, i, value):
        """
        Insert a value to a pose object (superclass method)

        :param i: element to insert value before
        :type i: int
        :param value: the value to insert
        :type value: SO2, SE2, SO3, SE3 instance
        :raises ValueError: incorrect type of inserted value

        Inserts the argument into the object's internal list of values.
        
        Examples::
            
            >>> x = SE3()
            >>> x.inert(0, SE3.Rx(0.1)) # insert at position 0 in the list
            >>> len(x)
            2
        """
        if not type(self) == type(value):
            raise ValueError("cant append different type of pose object")
        if len(value) > 1:
            raise ValueError("cant insert a pose sequence - must have len() == 1")
        super().insert(i, value.A)
        
    def pop(self):
        """
        Pop value of a pose object (superclass method)

        :return: the specific element of the pose
        :rtype: SO2, SE2, SO3, SE3 instance
        :raises IndexError: if there are no values to pop

        Removes the first pose value from the sequence in the pose object.
        
        Example::
            
            >>> x = SE3.Rx([0, math.pi/2, math.pi])
            >>> len(x)
            3
            >>> y = x.pop()
            >>> y
            SE3(array([[ 1.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00],
                       [ 0.0000000e+00, -1.0000000e+00, -1.2246468e-16,  0.0000000e+00],
                       [ 0.0000000e+00,  1.2246468e-16, -1.0000000e+00,  0.0000000e+00],
                       [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]]))
            >>> len(x)
            2
        """

        return self.__class__(super().pop())


# ------------------------------------------------------------------------ #

    # --------- compatibility methods

    def isrot(self):
        """
        Test if object belongs to SO(3) group (superclass method)

        :return: ``True`` if object is instance of SO3
        :rtype: bool

        For compatibility with Spatial Math Toolbox for MATLAB.
        In Python use ``isinstance(x, SO3)``.
        
        Example::
            
            >>> x = SO3()
            >>> x.isrot()
            True
            >>> x = SE3()
            >>> x.isrot()
            False
        """
        return type(self).__name__ == 'SO3'

    def isrot2(self):
        """
        Test if object belongs to SO(2) group (superclass method)

        :return: ``True`` if object is instance of SO2
        :rtype: bool

        For compatibility with Spatial Math Toolbox for MATLAB.
        In Python use ``isinstance(x, SO2)``.

        Example::
            
            >>> x = SO2()
            >>> x.isrot()
            True
            >>> x = SE2()
            >>> x.isrot()
            False
        """
        return type(self).__name__ == 'SO2'

    def ishom(self):
        """
        Test if object belongs to SE(3) group (superclass method)

        :return: ``True`` if object is instance of SE3
        :rtype: bool

        For compatibility with Spatial Math Toolbox for MATLAB.
        In Python use ``isinstance(x, SE3)``.
        
        Example::
            
            >>> x = SO3()
            >>> x.isrot()
            False
            >>> x = SE3()
            >>> x.isrot()
            True
        """
        return type(self).__name__ == 'SE3'

    def ishom2(self):
        """
        Test if object belongs to SE(2) group (superclass method)

        :return: ``True`` if object is instance of SE2
        :rtype: bool

        For compatibility with Spatial Math Toolbox for MATLAB.
        In Python use ``isinstance(x, SE2)``.
        
        Example::
            
            >>> x = SO2()
            >>> x.isrot()
            False
            >>> x = SE2()
            >>> x.isrot()
            True
        """
        return type(self).__name__ == 'SE2'
    
     #----------------------- functions

    def log(self):
        """
        Logarithm of pose (superclass method)

        :return: logarithm
        :rtype: numpy.ndarray
        :raises: ValueError
    
        An efficient closed-form solution of the matrix logarithm.
        
        =====  ======  ===============================
        Input         Output
        -----  ---------------------------------------
        Pose   Shape   Structure
        =====  ======  ===============================
        SO2    (2,2)   skew-symmetric
        SE2    (3,3)   augmented skew-symmetric
        SO3    (3,3)   skew-symmetric
        SE3    (4,4)   augmented skew-symmetric
        =====  ======  ===============================
        
        Example::

            >>> x = SE3.Rx(0.3)
            >>> y = x.log()
            >>> y
            array([[ 0. , -0. ,  0. ,  0. ],
                   [ 0. ,  0. , -0.3,  0. ],
                   [-0. ,  0.3,  0. ,  0. ],
                   [ 0. ,  0. ,  0. ,  0. ]])
            

        :seealso: :func:`~spatialmath.base.transforms2d.trlog2`, :func:`~spatialmath.base.transforms3d.trlog`
        """
        print('in log')
        if self.N == 2:
            log = [tr.trlog2(x) for x in self.data]
        else:
            log = [tr.trlog(x) for x in self.data]
        if len(log) == 1:
            return log[0]
        else:
            return log

    def interp(self, s=None, T0=None):
        """
        Interpolate pose (superclass method)
        
        :param T0: initial pose
        :type T0: SO2, SE2, SO3, SE3
        :param s: interpolation coefficient, range 0 to 1
        :type s: float or array_like
        :return: interpolated pose
        :rtype: SO2, SE2, SO3, SE3 instance
        
        - ``X.interp(s)`` interpolates the pose X between identity when s=0
          and X when s=1.

         ======  ======  ===========  ===============================
         len(X)  len(s)  len(result)  Result
         ======  ======  ===========  ===============================
         1       1       1            Y = interp(identity, X, s)
         M       1       M            Y[i] = interp(T0, X[i], s)
         1       M       M            Y[i] = interp(T0, X, s[i])
         ======  ======  ===========  ===============================

        Example::
            
            >>> x = SE3.Rx(0.3)
            >>> print(x.interp(0))
            SE3(array([[1., 0., 0., 0.],
                       [0., 1., 0., 0.],
                       [0., 0., 1., 0.],
                       [0., 0., 0., 1.]]))
            >>> print(x.interp(1))
            SE3(array([[ 1.        ,  0.        ,  0.        ,  0.        ],
                       [ 0.        ,  0.95533649, -0.29552021,  0.        ],
                       [ 0.        ,  0.29552021,  0.95533649,  0.        ],
                       [ 0.        ,  0.        ,  0.        ,  1.        ]]))
            >>> y = x.interp(x, np.linspace(0, 1, 10))
            >>> len(y)
            10
            >>> y[5]
            SE3(array([[ 1.        ,  0.        ,  0.        ,  0.        ],
                       [ 0.        ,  0.98614323, -0.16589613,  0.        ],
                       [ 0.        ,  0.16589613,  0.98614323,  0.        ],
                       [ 0.        ,  0.        ,  0.        ,  1.        ]]))
            
        Notes:
            
        #. For SO3 and SE3 rotation is interpolated using quaternion spherical linear interpolation (slerp).
    
        :seealso: :func:`~spatialmath.base.transforms3d.trinterp`, :func:`spatialmath.base.quaternions.slerp`, :func:`~spatialmath.base.transforms2d.trinterp2`
        """
        s = argcheck.getvector(s)
        if T0 is not None:
            assert len(T0) == 1, 'len(X0) must == 1'
            T0 = T0.A
            
        if self.N == 2:
            if len(s) > 1:
                assert len(self) == 1, 'if len(s) > 1, len(X) must == 1'
                return self.__class__([tr.trinterp2(self.A, T0, _s) for _s in s])
            else:
                assert len(s) == 1, 'if len(X) > 1, len(s) must == 1'
                return self.__class__([tr.trinterp2(x, T0, s) for x in self.data])
        elif self.N == 3:
            if len(s) > 1:
                assert len(self) == 1, 'if len(s) > 1, len(X) must == 1'
                return self.__class__([tr.trinterp(self.A, T1=T0, s=_s) for _s in s])
            else:
                assert len(s) == 1, 'if len(X) > 1, len(s) must == 1'
                return self.__class__([tr.trinterp(x, T1=T0, s=s) for x in self.data])
        
    
    def norm(self):
        """
        Normalize pose (superclass method)
        
        :return: pose
        :rtype: SO2, SE2, SO3, SE3 instance
    
        - ``X.norm()`` is an equivalent pose object but the rotational matrix 
          part of all values has been adjusted to ensure it is a proper orthogonal
          matrix rotation.
          
        Example::
            
            >>> x = SE3()
            >>> y = x.norm()
            >>> y
            SE3(array([[1., 0., 0., 0.],
                       [0., 1., 0., 0.],
                       [0., 0., 1., 0.],
                       [0., 0., 0., 1.]]))
    
        Notes:
            
        #. Only the direction of A vector (the z-axis) is unchanged.
        #. Used to prevent finite word length arithmetic causing transforms to 
           become 'unnormalized'.
           
        :seealso: :func:`~spatialmath.base.transforms3d.trnorm`, :func:`~spatialmath.base.transforms2d.trnorm2`
        """
        if self.N == 2:
            return self.__class__([tr.trnorm2(x) for x in self.data])
        else:
            return self.__class__([tr.trnorm(x) for x in self.data])

 

    # ----------------------- i/o stuff

    def printline(self, **kwargs):
        """
        Print pose as a single line (superclass method)
    
        :param label: text label to put at start of line
        :type label: str
        :param file: file to write formatted string to. [default, stdout]
        :type file: str
        :param fmt: conversion format for each number as used by ``format()``
        :type fmt: str
        :param unit: angular units: 'rad' [default], or 'deg'
        :type unit: str
        :return: optional formatted string
        :rtype: str
        
        For SO(3) or SE(3) also:
        
        :param orient: 3-angle convention to use
        :type orient: str
        
        - ``X.printline()`` print ``X`` in single-line format to ``stdout``, followed
          by a newline
        - ``X.printline(file=None)`` return a string containing ``X`` in 
          single-line format
        
        Example::
            
            >>> x=SE3.Rx(0.3)
            >>> x.printline()
            t =        0,        0,        0; rpy/zyx =       17,        0,        0 deg
        

        """
        if self.N == 2:
            tr.trprint2(self.A, **kwargs)
        else:
            tr.trprint(self.A, **kwargs)

    def __repr__(self):
        """
        Readable representation of pose (superclass method)
        
        :return: readable representation of the pose as a list of arrays
        :rtype: str
        
        Example::
            
            >>> x = SE3.Rx(0.3)
            >>> x
            SE3(array([[ 1.        ,  0.        ,  0.        ,  0.        ],
                       [ 0.        ,  0.95533649, -0.29552021,  0.        ],
                       [ 0.        ,  0.29552021,  0.95533649,  0.        ],
                       [ 0.        ,  0.        ,  0.        ,  1.        ]]))

        """
        name = type(self).__name__
        if len(self) ==  0:
            return name + '([])'
        elif len(self) == 1:
            # need to indent subsequent lines of the native repr string by 4 spaces
            return name + '(' + self.A.__repr__().replace('\n', '\n    ') + ')'
        else:
            # format this as a list of ndarrays
            return name + '([\n' + ',\n'.join([v.__repr__() for v in self.data]) + ' ])'

    def __str__(self):
        """
        Pretty string representation of pose (superclass method)

        :return: readable representation of the pose
        :rtype: str
        
        Convert the pose's matrix value to a simple grid of numbers.
        
        Example::
            
            >>> x = SE3.Rx(0.3)
            >>> print(x)
               1           0           0           0            
               0           0.955336   -0.29552     0            
               0           0.29552     0.955336    0            
               0           0           0           1 
        
        Notes:
            
            - By default, the output is colorised for an ANSI terminal console:
                
                * red: rotational elements
                * blue: translational elements
                * white: constant elements

        """
        return self._string(color=True)

    def _string(self, color=False, tol=10):
        """
        Pretty print the matrix value
        
        :param color: colorise the output, defaults to False
        :type color: bool, optional
        :param tol: zero values smaller than tol*eps, defaults to 10
        :type tol: float, optional
        :return: multiline matrix representation
        :rtype: str
        
        Convert a matrix to a simple grid of numbers with optional
        colorization for an ANSI terminal console:
                
                * red: rotational elements
                * blue: translational elements
                * white: constant elements
        
        Example::
            
            >>> x = SE3.Rx(0.3)
            >>> print(str(x))
               1           0           0           0            
               0           0.955336   -0.29552     0            
               0           0.29552     0.955336    0            
               0           0           0           1 

        """
        #print('in __str__')

        FG = lambda c: fg(c) if _color else ''
        BG = lambda c: bg(c) if _color else ''
        ATTR = lambda c: attr(c) if _color else ''

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
                        if tol > 0 and abs(element) < tol * _eps:
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
    
    # ----------------------- graphics
    
    def plot(self, *args, **kwargs):
        """
        Plot pose object as a coordinate frame (superclass method)
        
        :param `**kwargs`: plotting options
        
        - ``X.plot()`` displays the pose ``X`` as a coordinate frame in either
          2D or 3D axes.  There are many options, see the links below.

        Example::
            
            >>> X = SE3.Rx(0.3)
            >>> X.plot(frame='A', color='green')
    
        :seealso: :func:`~spatialmath.base.transforms3d.trplot`, :func:`~spatialmath.base.transforms2d.trplot2`
        """
        if self.N == 2:
            tr.trplot2(self.A, *args, **kwargs)
        else:
            tr.trplot(self.A, *args, **kwargs)
            
    def animate(self, *args, T0=None, **kwargs):
        """
        Plot pose object as an animated coordinate frame (superclass method)
        
        :param `**kwargs`: plotting options
        
        - ``X.plot()`` displays the pose ``X`` as a coordinate frame moving
          from the origin, or ``T0``, in either 2D or 3D axes.  There are 
          many options, see the links below.

        Example::
            
            >>> X = SE3.Rx(0.3)
            >>> X.animate(frame='A', color='green')

        :seealso: :func:`~spatialmath.base.transforms3d.tranimate`, :func:`~spatialmath.base.transforms2d.tranimate2`
        """
        if T0 is not None:
            T0 = T0.A
        if self.N == 2:
            tr.tranimate2(self.A, T0=T0, *args, **kwargs)
        else:
            tr.tranimate(self.A, T0=T0, *args, **kwargs)


# ------------------------------------------------------------------------ #

    #----------------------- arithmetic

    def __mul__(left, right):
        """
        Overloaded ``*`` operator (superclass method)

        :arg left: left multiplicand
        :arg right: right multiplicand
        :return: product
        :raises: ValueError

        Pose composition, scaling or vector transformation:
        
        - ``X * Y`` compounds the poses ``X`` and ``Y``
        - ``X * s`` performs elementwise multiplication of the elements of ``X`` by ``s``
        - ``s * X`` performs elementwise multiplication of the elements of ``X`` by ``s``
        - ``X * v`` linear transform of the vector ``v``

        ==============   ==============   ===========  ======================
                   Multiplicands                   Product
        -------------------------------   -----------------------------------
            left             right            type           operation
        ==============   ==============   ===========  ======================
        Pose             Pose             Pose         matrix product
        Pose             scalar           NxN matrix   element-wise product
        scalar           Pose             NxN matrix   element-wise product
        Pose             N-vector         N-vector     vector transform
        Pose             NxM matrix       NxM matrix   transform each column
        ==============   ==============   ===========  ======================
        
        Notes:
            
        #. Pose is ``SO2``, ``SE2``, ``SO3`` or ``SE3`` instance
        #. N is 2 for ``SO2``, ``SE2``; 3 for ``SO3`` or ``SE3``
        #. scalar x Pose is handled by ``__rmul__``
        #. scalar multiplication is commutative but the result is not a group
           operation so the result will be a matrix
        #. Any other input combinations result in a ValueError.
        
        For pose composition the ``left`` and ``right`` operands may be a sequence

        =========   ==========   ====  ================================
        len(left)   len(right)   len     operation
        =========   ==========   ====  ================================
         1          1             1    ``prod = left * right``
         1          M             M    ``prod[i] = left * right[i]``
         N          1             M    ``prod[i] = left[i] * right``
         M          M             M    ``prod[i] = left[i] * right[i]``
        =========   ==========   ====  ================================

        For vector transformation there are three cases
        
        =========  ===========  =====  ==========================
              Multiplicands             Product
        ----------------------  ---------------------------------
        len(left)  right.shape  shape  operation
        =========  ===========  =====  ==========================
        1          (N,)         (N,)   vector transformation
        M          (N,)         (N,M)  vector transformations
        1          (N,M)        (N,M)  column transformation
        =========  ===========  =====  ==========================
        
        Notes:
            
        #. for the ``SE2`` and ``SE3`` case the vectors are converted to homogeneous
           form, transformed, then converted back to Euclidean form.

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
        Overloaded ``*`` operator (superclass method)

        :arg left: left multiplicand
        :arg right: right multiplicand
        :return: product
        :raises: NotImplemented
        
        Left-multiplication by a scalar
        
        - ``s * X`` performs elementwise multiplication of the elements of ``X`` by ``s``

        Notes:
            
        #. For other left-operands return ``NotImplemented``.  Other classes
          such as ``Plucker`` and ``Twist`` implement left-multiplication by
          an ``SE33`` using their own ``__rmul__`` methods.
        
        """
        if isinstance(left, (int, np.int64, float, np.float64)):
            return right.__mul__(left)
        else:
            return NotImplemented

    def __imul__(left, right):
        """
        Overloaded ``*=`` operator (superclass method)

        :arg left: left multiplicand
        :arg right: right multiplicand
        :return: product
        :raises: ValueError

        - ``X *= Y`` compounds the poses ``X`` and ``Y`` and places the result in ``X``
        - ``X *= s`` performs elementwise multiplication of the elements of ``X``
          and ``s`` and places the result in ``X``

        :seealso: ``__mul__``
        """
        return left.__mul__(right)

    def __pow__(self, n):
        """
        Overloaded ``**`` operator (superclass method)
        
        :param n: pose
        :return: pose to the power n
        :type self: SO2, SE2, SO3, SE3

        Raise all elements of pose to the specified power.
        
        - ``X**n`` raise all values in ``X`` to the power ``n``
        """

        assert type(n) is int, 'exponent must be an int'
        return self.__class__([np.linalg.matrix_power(x, n) for x in self.data])

    # def __ipow__(self, n):
    #     return self.__pow__(n)

    def __truediv__(left, right):
        """
        Overloaded ``/`` operator (superclass method)
        
        :arg left: left multiplicand
        :arg right: right multiplicand
        :return: product
        :raises ValueError: for incompatible arguments
        :return: matrix
        :rtype: numpy ndarray
        
        Pose composition or scaling:
        
        - ``X / Y`` compounds the poses ``X`` and ``Y.inv()``
        - ``X / s`` performs elementwise multiplication of the elements of ``X`` by ``s``

        ==============   ==============   ===========  =========================
                   Multiplicands                   Quotient
        -------------------------------   --------------------------------------
            left             right            type           operation
        ==============   ==============   ===========  =========================
        Pose             Pose             Pose         matrix product by inverse
        Pose             scalar           NxN matrix   element-wise division
        ==============   ==============   ===========  =========================
        
        Notes:
            
        #. Pose is ``SO2``, ``SE2``, ``SO3`` or ``SE3`` instance
        #. N is 2 for ``SO2``, ``SE2``; 3 for ``SO3`` or ``SE3``
        #. scalar multiplication is not a group operation so the result will 
           be a matrix
        #. Any other input combinations result in a ValueError.
        
        For pose composition the ``left`` and ``right`` operands may be a sequence

        =========   ==========   ====  ================================
        len(left)   len(right)   len     operation
        =========   ==========   ====  ================================
         1          1             1    ``prod = left * right.inv()``
         1          M             M    ``prod[i] = left * right[i].inv()``
         N          1             M    ``prod[i] = left[i] * right.inv()``
         M          M             M    ``prod[i] = left[i] * right[i].inv()``
        =========   ==========   ====  ================================

        """
        if isinstance(left, right.__class__):
            return left.__class__(left._op2(right.inv(), lambda x, y: x @ y))
        elif isinstance(right, (int, np.int64, float, np.float64)):
            return left._op2(right, lambda x, y: x / y)
        else:
            raise ValueError('bad operands')

    # def __itruediv__(left, right):
    #     """
    #     Overloaded ``/=`` operator (superclass method)

    #     :arg left: left dividend
    #     :arg right: right divisor
    #     :return: quotient
    #     :raises: ValueError

    #     - ``X /= Y`` compounds the poses ``X`` and ``Y.inv()`` and places the result in ``X``
    #     - ``X /= s`` performs elementwise division of the elements of ``X`` by ``s``

    #     :seealso: ``__truediv__``
    #     """
    #     return left.__truediv__(right)

    def __add__(left, right):
        """
        Overloaded ``+`` operator (superclass method)
        
        :arg left: left addend
        :arg right: right addend
        :return: sum
        :raises ValueError: for incompatible arguments
        :return: matrix
        :rtype: numpy ndarray, shape=(N,N)
        
        Add elements of two poses.  This is not a group operation so the
        result is a matrix not a pose class.
                
        - ``X + Y`` is the element-wise sum of the matrix value of ``X`` and ``Y``
        - ``X + s`` is the element-wise sum of the matrix value of ``X`` and ``s``
        - ``s + X`` is the element-wise sum of the matrix value of ``s`` and ``X``

        ==============   ==============   ===========  ========================
                   Operands                   Sum
        -------------------------------   -------------------------------------
            left             right            type           operation
        ==============   ==============   ===========  ========================
        Pose             Pose             NxN matrix   element-wise matrix sum
        Pose             scalar           NxN matrix   element-wise sum
        scalar           Pose             NxN matrix   element-wise sum
        ==============   ==============   ===========  ========================
        
        Notes:
            
        #. Pose is ``SO2``, ``SE2``, ``SO3`` or ``SE3`` instance
        #. N is 2 for ``SO2``, ``SE2``; 3 for ``SO3`` or ``SE3``
        #. scalar + Pose is handled by ``__radd__``
        #. scalar addition is commutative
        #. Any other input combinations result in a ValueError.
        
        For pose addition the ``left`` and ``right`` operands may be a sequence which
        results in the result being a sequence:
            
        =========   ==========   ====  ================================
        len(left)   len(right)   len     operation
        =========   ==========   ====  ================================
         1          1             1    ``prod = left + right``
         1          M             M    ``prod[i] = left + right[i]``
         N          1             M    ``prod[i] = left[i] + right``
         M          M             M    ``prod[i] = left[i] + right[i]``
        =========   ==========   ====  ================================

        """
        # results is not in the group, return an array, not a class
        return left._op2(right, lambda x, y: x + y)

    def __radd__(left, right):
        """
        Overloaded ``+`` operator (superclass method)

        :arg left: left addend
        :arg right: right addend
        :return: sum
        :raises ValueError: for incompatible arguments
        
        Left-addition by a scalar
        
        - ``s + X`` performs elementwise addition of the elements of ``X`` and ``s``
        
        """
        return left.__add__(right)

    # def __iadd__(left, right):
    #     return left.__add__(right)

    def __sub__(left, right):
        """
        Overloaded ``-`` operator (superclass method)
        
        :arg left: left minuend
        :arg right: right subtrahend
        :return: difference
        :raises ValueError: for incompatible arguments
        :return: matrix
        :rtype: numpy ndarray, shape=(N,N)
        
        Subtract elements of two poses.  This is not a group operation so the
        result is a matrix not a pose class.
                
        - ``X - Y`` is the element-wise difference of the matrix value of ``X`` and ``Y``
        - ``X - s`` is the element-wise difference of the matrix value of ``X`` and ``s``
        - ``s - X`` is the element-wise difference of ``s`` and the matrix value of ``X``

        ==============   ==============   ===========  ==============================
                   Operands                   Sum
        -------------------------------   -------------------------------------------
            left             right            type           operation
        ==============   ==============   ===========  ==============================
        Pose             Pose             NxN matrix   element-wise matrix difference
        Pose             scalar           NxN matrix   element-wise sum
        scalar           Pose             NxN matrix   element-wise sum
        ==============   ==============   ===========  ==============================
        
        Notes:
            
        #. Pose is ``SO2``, ``SE2``, ``SO3`` or ``SE3`` instance
        #. N is 2 for ``SO2``, ``SE2``; 3 for ``SO3`` or ``SE3``
        #. scalar - Pose is handled by ``__rsub__``
        #. Any other input combinations result in a ValueError.
        
        For pose addition the ``left`` and ``right`` operands may be a sequence which
        results in the result being a sequence:

        =========   ==========   ====  ================================
        len(left)   len(right)   len     operation
        =========   ==========   ====  ================================
         1          1             1    ``prod = left - right``
         1          M             M    ``prod[i] = left - right[i]``
         N          1             M    ``prod[i] = left[i] - right``
         M          M             M    ``prod[i] = left[i]  right[i]``
        =========   ==========   ====  ================================
        """

        # results is not in the group, return an array, not a class
        # TODO allow class +/- a conformant array
        return left._op2(right, lambda x, y: x - y)

    def __rsub__(left, right):
        """
        Overloaded ``-`` operator (superclass method)

        :arg left: left minuend
        :arg right: right subtrahend
        :return: difference
        :raises ValueError: for incompatible arguments
        
        Left-addition by a scalar
        
        - ``s + X`` performs elementwise addition of the elements of ``X`` and ``s``
        
        """
        return -left.__sub__(right)

    # def __isub__(left, right):
    #     return left.__sub__(right)

    def __eq__(left, right):
        """
        Overloaded ``==`` operator (superclass method)
        
        :param left: left side of comparison
        :type self: SO2, SE2, SO3, SE3
        :param right: right side of comparison
        :type self: SO2, SE2, SO3, SE3
        :return: poses are equal
        :rtype: bool
        
        Test two poses for equality
        
        - ``X == Y`` is true of the poses are of the same type and numerically
          equal.

        If either operand contains a sequence the results is a sequence 
        according to:
        
        =========   ==========   ====  ================================
        len(left)   len(right)   len     operation
        =========   ==========   ====  ================================
         1          1             1    ``ret = left == right``
         1          M             M    ``ret[i] = left == right[i]``
         N          1             M    ``ret[i] = left[i] == right``
         M          M             M    ``ret[i] = left[i] == right[i]``
        =========   ==========   ====  ================================

        """
        assert type(left) == type(right), 'operands to == are of different types'
        return left._op2(right, lambda x, y: np.allclose(x, y))

    def __ne__(left, right):
        """
        Overloaded ``!=`` operator
        
        :param left: left side of comparison
        :type self: SO2, SE2, SO3, SE3
        :param right: right side of comparison
        :type self: SO2, SE2, SO3, SE3
        :return: poses are not equal
        :rtype: bool
        
        Test two poses for inequality
        
        - ``X == Y`` is true of the poses are of the same type but not numerically
          equal.
          
        If either operand contains a sequence the results is a sequence 
        according to:
        
        =========   ==========   ====  ================================
        len(left)   len(right)   len     operation
        =========   ==========   ====  ================================
         1          1             1    ``ret = left != right``
         1          M             M    ``ret[i] = left != right[i]``
         N          1             M    ``ret[i] = left[i] != right``
         M          M             M    ``ret[i] = left[i] != right[i]``
        =========   ==========   ====  ================================

        """
        return [not x for x in self == right]

    def _op2(left, right, op): 
        """
        Perform binary operation
        
        :param left: left side of comparison
        :type self: SO2, SE2, SO3, SE3
        :param right: right side of comparison
        :type self: SO2, SE2, SO3, SE3
        :param op: binary operation
        :type op: callable
        :raises ValueError: arguments are not compatible
        :return: list of matrices
        :rtype: list
        
        Peform a binary operation on a pair of operands.  If either operand
        contains a sequence the results is a sequence accordinging to this
        truth table.

        =========   ==========   ====  ================================
        len(left)   len(right)   len     operation
        =========   ==========   ====  ================================
         1          1             1    ``ret = op(left, right)``
         1          M             M    ``ret[i] = op(left, right[i])``
         N          1             M    ``ret[i] = op(left[i], right)``
         M          M             M    ``ret[i] = op(left[i], right[i])``
        =========   ==========   ====  ================================

        """

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
        elif isinstance(right, (float, int, np.float64, np.int64)) or (isinstance(right, np.ndarray) and right.shape == left.shape):
            # class by matrix
            if len(left) == 1:
                return op(left.A, right)
            else:
                return [op(x, right) for x in left.A]



    
    
class SMTwist(UserList):
    """
    Superclass for 2D and 3D twist objects

    Subclasses are:

    - ``Twist2`` representing rigid-body motion in 2D as a 3-vector
    - ``Twist`` representing rigid-body motion in 3D as a 6-vector

    A twist is the unique elements of the logarithm of the corresponding SE(N)
    matrix.
    
    Arithmetic operators are overloaded but the operation they perform depend
    on the types of the operands.  For example:

    - ``*`` will compose two instances of the same subclass, and the result will be
      an instance of the same subclass, since this is a group operator.

    These classes all inherit from ``UserList`` which enables them to 
    represent a sequence of values, ie. an ``Twist`` instance can contain
    a sequence of twists.  Most of the Python ``list`` operators
    are applicable::

        >>> x = Twist()  # new instance with zero value
        >>> len(x)     # it is a sequence of one value
        1
        >>> x.append(x)  # append to itself
        >>> len(x)       # it is a sequence of two values
        2
        >>> x[1]         # the element has a 4x4 matrix value
        SE3([
        array([[1., 0., 0., 0.],
               [0., 1., 0., 0.],
               [0., 0., 1., 0.],
            [0., 0., 0., 1.]]) ])
        >>> x[1] = SE3.Rx(0.3)  # set an elements of the sequence
        >>> x.reverse()         # reverse the elements in the sequence
        >>> del x[1]            # delete an element

    """
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
        Construct an empty twist object (superclass method)
        
        :param cls: The twist subclass
        :type cls: type
        :return: a twist object with zero values
        :rtype: Twist or Twist2 instance

        Example::
            
            >>> x = Twist.Empty()
            >>> len(x)
            0
        """
        X = cls()
        X.data = []
        return X
    
    @property
    def S(self):
        """
        Twist as a vector (superclass property)
        
        :return: Twist vector
        :rtype: numpy.ndarray, shape=(N,)
        
        - ``X.S`` is a 3-vector if X is a ``Twist2`` instance, and a 6-vector if
          X is a ``Twist`` instance.

        Notes::
            
            
        - the vector is the unique elements of the se(N) representation
        - the vector is sometimes referred to as the twist coordinate vector.
        - if ``len(X)`` > 1 then return a list of vectors.
        """
        # get the underlying numpy array
        if len(self.data) == 1:
            return self.data[0]
        else:
            return self.data
        
    @property
    def isprismatic(self):
        """
        Test for prismatic twist (superclass property)
        
        :return: If twist is purely prismatic
        :rtype: book
        
        Example::
            
            >>> x = Twist.R([1,2,3], [4,5,6])
            >>> x.isprismatic
            False

        """
        if len(self) == 1:
            return tr.iszerovec(self.w)
        else:
            return [tr.iszerovec(x.w) for x in self.data]

    @property
    def unit(self):
        """
        Unit twist

        TW.unit() is a Twist object representing a unit aligned with the Twist
        TW.
        """
        if tr.iszerovec(self.w):
            # rotational twist
            return Twist(self.S / tr.norm(S.w))
        else:
            # prismatic twist
            return Twist(tr.unitvec(self.v), [0, 0, 0])
    
    @property
    def isunit(self):
        """
        Test for unit twist (superclass property)
        
        :return: If twist is a unit-twist
        :rtype: bool
        """
        if len(self) == 1:
            return tr.isunittwist(self.S)
        else:
            return [tr.isunittwist(x) for x in self.data]


    def __getitem__(self, i):
        """
        Access value of a twist object (superclass method)

        :param i: index of element to return
        :type i: int
        :return: the specific element of the twist
        :rtype: Twist or Twist2 instance
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
        if isinstance(i, slice):
            return self.__class__([self.data[k] for k in range(i.start or 0, i.stop or len(self), i.step or 1)], check=False)
        else:
            return self.__class__(self.data[i], check=False)
        
    def __setitem__(self, i, value):
        """
        Assign a value to a twist object (superclass method)
        
        :param i: index of element to assign to
        :type i: int
        :param value: the value to insert
        :type value: Twist or Twist2 instance
        :raises ValueError: incorrect type of assigned value

        Assign the argument to an element of the object's internal list of values.
        This supports the assignement operator, for example::
            
            >>> x = SE3([SE3() for i in range(10)]) # sequence of ten identity values
            >>> len(x)
            10
            >>> x[3] = SE3.Rx(0.2)   # assign to position 3 in the list
        """
        if not type(self) == type(value):
            raise ValueError("cant append different type of pose object")
        if len(value) > 1:
            raise ValueError("cant insert a pose sequence - must have len() == 1")
        self.data[i] = value.A
    
    def append(self, x):
        """
        Append a twist object
        
        :param x: A twist subclass
        :type x: subclass
        :raises ValueError: incorrect type of appended object

        Appends the argument to the object's internal list.
        
        Examples::
            
            >>> x = Twist()
            >>> len(x)
            1
            >>> x.append(Twist())
            >>> len(x)
            2
        """
        #print('in append method')
        if not type(self) == type(x):
            raise ValueError("cant append different type of pose object")
        if len(x) > 1:
            raise ValueError("cant append a pose sequence - use extend")
        super().append(x.S)
        
    def pop(self):
        """
        Pop value of a pose object (superclass method)

        :return: the specific element of the pose
        :rtype: SO2, SE2, SO3, SE3 instance
        :raises IndexError: if there are no values to pop

        Removes the first pose value from the sequence in the pose object.
        
        Example::
            
            >>> x = SE3.Rx([0, math.pi/2, math.pi])
            >>> len(x)
            3
            >>> y = x.pop()
            >>> y
            SE3(array([[ 1.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00],
                       [ 0.0000000e+00, -1.0000000e+00, -1.2246468e-16,  0.0000000e+00],
                       [ 0.0000000e+00,  1.2246468e-16, -1.0000000e+00,  0.0000000e+00],
                       [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]]))
            >>> len(x)
            2
        """

        return self.__class__(super().pop())
    
    def insert(self, i, value):
        """
        Insert a value to a pose object (superclass method)

        :param i: element to insert value before
        :type i: int
        :param value: the value to insert
        :type value: SO2, SE2, SO3, SE3 instance
        :raises ValueError: incorrect type of inserted value

        Inserts the argument into the object's internal list of values.
        
        Examples::
            
            >>> x = SE3()
            >>> x.inert(0, SE3.Rx(0.1)) # insert at position 0 in the list
            >>> len(x)
            2
        """
        if not type(self) == type(value):
            raise ValueError("cant append different type of pose object")
        if len(value) > 1:
            raise ValueError("cant insert a pose sequence - must have len() == 1")
        super().insert(i, value.A)
        

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


