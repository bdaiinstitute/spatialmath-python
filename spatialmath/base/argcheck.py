#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utilities functions for testing passed arguments to spatialmath.
"""

# pylint: disable=invalid-name

import math
import numpy as np
from spatialmath.base import symbolic as sym

# valid scalar types
_scalartypes = (int, np.int64, float, np.float64) + sym.symtype

def isscalar(x):
    """
    Test if argument is a real scalar

    :param x: value to test
    :return: True if scalar
    :rtype: bool

    ``isscalar(x)`` is ``True`` if ``x`` is a Python or numPy int or real float.
    """
    return isinstance(x, _scalartypes)


def assertmatrix(m, shape=None):
    """
    Assert that argument is a 2D matrix

    :param m: value to test
    :param shape: required shape
    :type shape: 2-tuple
    :raises: AssertionError

    - ``assertsmatrix(A, (2,3))`` raises an ``ValueError`` if ``A`` is not
      a 2x3 numPy ndarray, ie. shape=(2,3).
    - ``assertsmatrix(A, (2,None))`` raises an ``ValueError`` if ``A`` is not
      a 2xN numPy ndarray, ie. shape=(2,N).
    - ``assertsmatrix(A, (None,3))`` raises an ``ValueError`` if ``A`` is not
      a Nx3 numPy ndarray, ie. shape=(N,3).

    :seealso: :func:`ismatrix`
    """

    if not isinstance(m, np.ndarray):
        raise TypeError("input must be a numPy ndarray")
    if m.dtype.kind == 'c':
        raise TypeError("input must be a real numPy ndarray")
    if shape is not None:
        if len(shape) != len(m.shape):
            raise ValueError("incorrect scalarber of matrix dimensions, expecting {}, got {}".format(shape, m.shape))
        if shape[0] is not None and shape[0] > 0 and shape[0] != m.shape[0]:
            raise ValueError("incorrect matrix dimensions, expecting {}, got {}".format(shape, m.shape))
        if len(shape) > 1 and shape[1] is not None and shape[1] > 0 and shape[1] != m.shape[1]:
            raise ValueError("incorrect matrix dimensions, expecting {}, got {}".format(shape, m.shape))

def ismatrix(m, shape):
    """
    Test if argument is a real 2D matrix

    :param m: value to test
    :param shape: required shape
    :type shape: 2-tuple
    :return: True if value is of specified shape
    :rtype: bool

    Tests if the argument is a 2D matrix with a specified shape ``shape`` but
    this can indicate an unspecified (wildcard) dimension, for example:

    - ``issmatrix(A, (2,3))`` is ``True`` if ``A`` is a 2x3 numPy ndarray.
    - ``issmatrix(A, (2,None))`` is ``True`` if ``A`` is a 2xN numPy ndarray.
    - ``issmatrix(A, (None,3))`` is ``True`` if ``A`` is an Nx3 numPy ndarray.

    .. notes::
       - Unlike ``verifymatrix`` this function:
         - checks the argument is real valued
         - allows the shape to have an unspecified dimension

    :seealso: getmatrix, verifymatrix
    """
    if not isinstance(m, np.ndarray):
        return False
    if m.dtype.kind == 'c':
        return False
    if len(shape) != len(m.shape):
        return False
    if shape[0] is not None and shape[0] > 0 and shape[0] != m.shape[0]:
        return False
    if shape[1] is not None and shape[1] > 0 and shape[1] != m.shape[1]:
        return False
    return True

def getmatrix(m, shape):
    r"""
    Converrt argument to 2D array

    :param m: input value
    :type m: 2D NumPy ndarray, 1D array-like or scalar
    :param shape: shape of returned matrix
    :type shape: 2-tupe
    :raises ValueError: if ``m`` is inconsistent with ``shape``
    :raises TypeError: if ``m`` is not required type
    :return: a 2D array
    :rtype: NumPy ndarray

    ``getmatrix(m, shape)`` is a 2D matrix with shape ``shape`` formed from
    ``m`` which can be a 2D array, 1D array-like or a scalar.

    .. notes::
       - If ``m`` is a 2D array its shape is compared to ``shape`` - a 2-tuple
         where ``None`` stands for unspecified, ie. ``(None, 2)`` will match
         any array where the second dimension is 2.
       - If ``m`` is a 1D array its shape is checked to see if it can be
         reshaped to ``shape``.  A n-array could be reshaped as (n,1) or (1,n)
         or any other shape with the correct number of elements.  A value of
         ``None`` in the shape stands for unspecified, ie. ``(None, 2)`` will 
         attempt to reshape ``m`` as an array with shape (k,2) where :math:`k \times 2 \eq n`.
       - If ``m`` is a scalar, return an array of shape (1,1)

    :seealso: ismatrix, verifymatrix
    """
    if isinstance(m, np.ndarray) and len(m.shape) == 2:
        # passed a 2D array
        mshape = m.shape

        if (shape[0] is None or shape[0] == mshape[0]) and (shape[1] is None or shape[1] == mshape[1]):
                return m
        else:
            raise ValueError(f"expecting {shape} but got {mshape}")

    elif isvector(m):
            # passed a 1D array
            m = getvector(m)
            if shape[0] is not None and shape[1] is not None:
                if len(m) == np.prod(shape):
                    return m.reshape(shape)
                else:
                    raise ValueError('array cannot be reshaped')
            elif shape[0] is not None and shape[1] is None:
                return m.reshape((shape[0], -1))
            elif shape[0] is None and shape[1] is not None:
                return m.reshape((-1, shape[1]))
            else:
                return m.reshape((1, -1))

    else:
        raise TypeError('argument must be scalar or ndarray')
  

def verifymatrix(m, shape):
    """
    Test if argument is array of specified type

    :param m: value to be tested
    :param shape: desired shape of value
    :type shape: 2-tuple
    :raises TypeError: argument is not a NumPy array
    :raises ValueError: argument has incorrect shape

    Raises an exception if the argument ``m`` is not a NumPy array of the 
    specified shape.

    :seealso: getmatrix, ismatrix
    """
    if not isinstance(m, np.ndarray):
        raise TypeError("input must be a numPy ndarray")

    if not m.shape == shape:
        raise ValueError("incorrect matrix dimensions, "
                         "expecting {0}".format(shape))

# and not np.iscomplex(m) checks every element, would need to be not np.any(np.iscomplex(m)) which seems expensive


def getvector(v, dim=None, out='array', dtype=np.float64):
    """
    Return a vector value

    :param v: passed vector
    :param dim: required dimension, or None if any length is ok
    :type dim: int or None
    :param out: output format, default is 'array'
    :type out: str
    :param dtype: datatype for numPy array return (default np.float64)
    :type dtype: numPy type
    :return: vector value in specified format

    The passed vector can be any of:

    - Python native list or tuple
    - numPy 1D array, ie. shape=(N,)
    - numPy 2D array with a singleton dimension, ie. shape=(1,N) or (N,1)

    The returned vector will be in the format specified by ``out``:

    ==========  ===============================================
    format      return type
    ==========  ===============================================
    'sequence'  Python list, or tuple if a tuple was passed in
    'array'     1D numPy array, shape=(N,)
    'row'       row vector, a 2D numPy array, shape=(1,N)
    'col'       column vector, 2D numPy array, shape=(N,1)
    ==========  ===============================================

    For 'array', 'row' or 'col' the numPy dtype defaults to ``np.float64`` but
    can be overriden using the ``dtype`` argument.
    """
    dt = dtype

    if isinstance(v, _scalartypes):  # handle scalar case
        v = [v]

    if isinstance(v, (list, tuple)):
        # list or tuple was passed in

        if sym.issymbol(v):
            dt = None
            
        if dim is not None and v and len(v) != dim:
            raise ValueError("incorrect vector length")
        if out == 'sequence':
            return v
        elif out == 'array':
            return np.array(v, dtype=dt)
        elif out == 'row':
            return np.array(v, dtype=dt).reshape(1, -1)
        elif out == 'col':
            return np.array(v, dtype=dt).reshape(-1, 1)
        else:
            raise ValueError("invalid output specifier")

    elif isinstance(v, np.ndarray):
        s = v.shape
        if dim is not None:
            if not (s == (dim,) or s == (1, dim) or s == (dim, 1)):
                raise ValueError("incorrect vector length: expected {}, got {}".format(dim, s))

        v = v.flatten()

        if v.dtype.kind != 'O':
            dt = dtype

        if out == 'sequence':
            return list(v.flatten())
        elif out == 'array':
            return v.astype(dt)
        elif out == 'row':
            return v.astype(dt).reshape(1, -1)
        elif out == 'col':
            return v.astype(dt).reshape(-1, 1)
        else:
            raise ValueError("invalid output specifier")
    else:
        raise TypeError("invalid input type")


def assertvector(v, dim):
    """
    Assert that argument is a real vector

    :param v: passed vector
    :param dim: required dimension, or None if any length is ok
    :type dim: int or None
    :raises: AssertionError

    - ``assertvector(vec, N)`` raises an ``AssertionError`` if ``vec`` is not
      an N-element vector.

    :seealso: :func:`isvector`
    """
    assert isvector(v, dim)


def isvector(v, dim=None):
    """
    Test if argument is a real vector

    :param v: value to test
    :param dim: required dimension, or None if any length is ok
    :type dim: int
    :return: True if is vector of specified length
    :rtype: bool

    - ``isvector(vec, N)`` is ``True`` if ``vec`` is an N-element vector.

    A valid vector can be any of:

    - a Python native int or float, a 1-vector
    - Python native list or tuple
    - numPy 1D array, ie. shape=(N,)
    - numPy 2D array with a singleton dimension, ie. shape=(1,N) or (N,1)

    Examples::

        >>> isvector([1,2])
        True
        >>> isvector((1,2))
        True
        >>> isvector(np.r_[1,2,3])
        True
        >>> isvector(1)
        True
        >>> isvector([1,2], 3)
        False
    """
    if isinstance(v, (list, tuple)) and (dim is None or len(v) == dim) \
       and all(map(lambda x: isinstance(x, _scalartypes), v)):
        return True  # list or tuple

    if isinstance(v, np.ndarray):
        s = v.shape
        if dim is None:
            return (len(s) == 1 and s[0] > 0) or (s[0] == 1 and s[1] > 0) \
                   or (s[0] > 0 and s[1] == 1)
        else:
            return s == (dim,) or s == (1, dim) or s == (dim, 1)

    if (dim is None or dim == 1) and isinstance(v, _scalartypes):
        return True

    return False


def getunit(v, unit):
    """
    Convert value according to angular units

    :param v: the value in radians or degrees
    :type v: float or np.ndarray
    :param unit: the angular unit, "rad" or "deg"
    :type unit: str
    :return: the converted value in radians
    :rtype: float

    Examples::

        >>> getunit(1.5, 'rad')
        1.5
        >>> getunit(90, 'deg')
        1.5707963267948966
        >>> getunit(np.r_[0.5, 1], 'rad')
        array([0.5, 1. ])
        >>> getunit(np.r_[90, 180], 'deg')
        array([1.57079633, 3.14159265])
    """

    if unit == "rad":
        return v
    elif unit == "deg":
        if isinstance(v, np.ndarray) or np.isscalar(v):
            return v * math.pi / 180
        else:
            return [x * math.pi / 180 for x in v]
    else:
        raise ValueError("invalid angular units")


def isnumberlist(x):
    """
    Test if argument is a list of scalars

    :param x: the value to test
    :return: True if the argument is a list of real scalars
    :rtype: bool

    ``isscalarberlist(x)`` is ``True`` if ``x```` is a list of scalars.

    Examples::

        >>> isscalarberlist((1,2,3))
        True
        >>> isscalarberlist([1.1, 2.2, 3.3])
        True
        >>> isscalarberlist(1)
        False
        >>> isscalarberlist(np.r_[1,2])
        False
    """

    return isinstance(x, (list, tuple)) and len(x) > 0 \
           and all(map(lambda x: isinstance(x, _scalartypes), x))


def isvectorlist(x, n):
    """
    Test if argument is a list of vectors

    :param x: the value to test
    :return: True if the argument is a list of n-vectors
    :rtype: bool

    ``isvectorlist(x, n)`` is ``True`` if ``x```` is a list or tuple of
    1D numPy arrays of shape=(n,).


    Examples::

        >>> isvectorlist([np.r_[1,2], np.r_[3,4], np.r_[5,6]], 2)
        True
        >>> isvectorlist((np.r_[1,2], np.r_[3,4], np.r_[5,6]), 2)
        True
        >>> isvectorlist([(1,2), (3,4), (5,6)], 2)
        False
        >>> isvectorlist([np.r_[1,2], np.r_[3,4], np.r_[5,6,7]], 2)
        False

    """
    return isinstance(x, (list, tuple)) and len(x) > 0 \
           and all(map(lambda x: isinstance(x, np.ndarray) and x.shape == (n,), x))


if __name__ == '__main__':
    import pathlib
    
    exec(open(pathlib.Path(__file__).parent.absolute() / "test_argcheck.py").read())  # pylint: disable=exec-used
