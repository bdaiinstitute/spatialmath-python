# for Python <= 3.8

from typing import (
    overload,
    Union,
    List,
    Tuple,
    Type,
    TextIO,
    Any,
    Callable,
    Optional,
    Iterator,
)
from typing_extensions import Literal as L
from typing_extensions import Self

# array like

# these are input to many functions in spatialmath.base, and can be a list, tuple or
# ndarray.  The elements are generally float, but some functions accept symbolic
# arguments as well, which leads to a NumPy array with dtype=object
#
# The variants like ArrayLike2 indicate that a list, tuple or ndarray of length 2 is
# expected.  Static checking of tuple length is possible but not a lists. This might be
# possible in future versions of Python, but for now it is a hint to the coder about
# what is expected

from numpy.typing import DTypeLike, NDArray  # , ArrayLike

from typing import cast

# from typing import TypeVar
# NDArray = TypeVar('NDArray')
import numpy as np


ArrayLike = Union[float, List[float], Tuple[float, ...], NDArray]
ArrayLikePure = Union[List[float], Tuple[float, ...], NDArray]
ArrayLike2 = Union[List, Tuple[float, float], NDArray]
ArrayLike3 = Union[List, Tuple[float, float, float], NDArray]
ArrayLike4 = Union[List, Tuple[float, float, float, float], NDArray]
ArrayLike6 = Union[List, Tuple[float, float, float, float, float, float], NDArray]

# real vectors
R1 = NDArray[np.floating]  # R^1
R2 = NDArray[np.floating]  # R^2
R3 = NDArray[np.floating]  # R^3
R4 = NDArray[np.floating]  # R^4
R6 = NDArray[np.floating]  # R^6
R8 = NDArray[np.floating]  # R^8

# real matrices
R1x1 = NDArray  # R^{1x1} matrix
R2x2 = NDArray  # R^{3x3} matrix
R3x3 = NDArray  # R^{3x3} matrix
R4x4 = NDArray  # R^{4x4} matrix
R6x6 = NDArray  # R^{6x6} matrix
R8x8 = NDArray  # R^{8x8} matrix

R1x3 = NDArray  # R^{1x3} row vector
R3x1 = NDArray  # R^{3x1} column vector
R1x2 = NDArray  # R^{1x2} row vector
R2x1 = NDArray  # R^{2x1} column vector

Points2 = NDArray  # R^{2xN} matrix
Points3 = NDArray  # R^{2xN} matrix

RNx3 = NDArray  # R^{Nx3} matrix


# Lie group elements
SO2Array = NDArray  # SO(2) rotation matrix
SE2Array = NDArray  # SE(2) rigid-body transform
SO3Array = NDArray  # SO(3) rotation matrix
SE3Array = NDArray  # SE(3) rigid-body transform

# Lie algebra elements
so2Array = NDArray  # so(2) Lie algebra of SO(2), skew-symmetrix matrix
se2Array = NDArray  # se(2) Lie algebra of SE(2), augmented skew-symmetrix matrix
so3Array = NDArray  # so(3) Lie algebra of SO(3), skew-symmetrix matrix
se3Array = NDArray  # se(3) Lie algebra of SE(3), augmented skew-symmetrix matrix

# quaternion arrays
QuaternionArray = NDArray
UnitQuaternionArray = NDArray

Rn = Union[R2, R3]

SOnArray = Union[SO2Array, SO3Array]
SEnArray = Union[SE2Array, SE3Array]

sonArray = Union[so2Array, so3Array]
senArray = Union[se2Array, se3Array]

# __all__ = [
#     overload,
#     Union,
#     List,
#     Tuple,
#     Type,
#     TextIO,
#     Any,
#     Callable,
#     Optional,
#     Iterator,
#     ArrayLike,
#     ArrayLike2,
#     ArrayLike3,
#     ArrayLike4,
#     ArrayLike6,
#     # real vectors
#     R2,
#     R3,
#     R4,
#     R6,
#     R8,
#     # real matrices
#     R2x2,
#     R3x3,
#     R4x4,
#     R6x6,
#     R8x8,
#     R1x3,
#     R3x1,
#     R1x2,
#     R2x1,
#     Points2,
#     Points3,
#     RNx3,
#     # Lie group elements
#     SO2Array,
#     SE2Array,
#     SO3Array,
#     SE3Array,
#     # Lie algebra elements
#     so2Array,
#     se2Array,
#     so3Array,
#     se3Array,
#     # quaternion arrays
#     QuaternionArray,
#     UnitQuaternionArray,
#     Rn,
#     SOnArray,
#     SEnArray,
#     sonArray,
#     senArray,
# ]
Color = Union[str, ArrayLike3]
