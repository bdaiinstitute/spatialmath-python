from typing import overload, Union, List, Tuple, Type, TextIO, Any, Callable, Optional, Literal as L
from numpy import ndarray, dtype, floating
from numpy.typing import DTypeLike

ArrayLike = Union[float,List,Tuple,ndarray[Any, dtype[floating]]]
R3 = ndarray[Tuple[L[3,]], dtype[floating]]  # R^3
R6 = ndarray[Tuple[L[6,]], dtype[floating]]  # R^6
SO3Array = ndarray[Tuple[L[3,3]], dtype[floating]]  # SO(3) rotation matrix
SE3Array = ndarray[Tuple[L[4,4]], dtype[floating]]  # SE(3) rigid-body transform
so3Array = ndarray[Tuple[L[3,3]], dtype[floating]]  # so(3) Lie algebra of SO(3), skew-symmetrix matrix
se3Array = ndarray[Tuple[L[4,4]], dtype[floating]]  # se(3) Lie algebra of SE(3), augmented skew-symmetrix matrix
R4x4 = ndarray[Tuple[L[4,4]], dtype[floating]]  # R^{4x4} matrix
R6x6 = ndarray[Tuple[L[6,6]], dtype[floating]]  # R^{6x6} matrix
R3x3 = ndarray[Tuple[L[3,3]], dtype[floating]]  # R^{3x3} matrix
R1x3 = ndarray[Tuple[L[1,3]], dtype[floating]]  # R^{1x3} row vector
R3x1 = ndarray[Tuple[L[3,1]], dtype[floating]]  # R^{3x1} column vector

R3x = Union[List,Tuple[float,float,float],R3,R3x1,R1x3]  # various ways to represent R^3 for input

R2 = ndarray[Any, dtype[floating]]  # R^6
SO2Array = ndarray[Tuple[L[2,2]], dtype[floating]]  # SO(3) rotation matrix
SE2Array = ndarray[Tuple[L[3,3]], dtype[floating]]  # SE(3) rigid-body transform
so2Array = ndarray[Tuple[L[2,2]], dtype[floating]]  # so(3) Lie algebra of SO(3), skew-symmetrix matrix
se2Array = ndarray[Tuple[L[3,3]], dtype[floating]]  # se(3) Lie algebra of SE(3), augmented skew-symmetrix matrix

R1x2 = ndarray[Tuple[L[1,2]], dtype[floating]]  # R^{1x2} row vector
R2x1 = ndarray[Tuple[L[2,1]], dtype[floating]]  # R^{2x1} column vector
R2x = Union[List,Tuple[float,float],R2,R2x1,R1x2]  # various ways to represent R^2 for input

# from typing import overload, Union, List, Tuple, TextIO, Any, Optional #, TypeGuard for 3.10
# # Array2 = Union[NDArray[(2,),np.dtype[np.floating]],np.ndarray[(2,1),np.dtype[np.floating]],np.ndarray[(1,2),np.dtype[np.floating]]]
# # Array3 = Union[np.ndarray[(3,),np.dtype[np.floating]],np.ndarray[(3,1),np.dtype[np.floating]],np.ndarray[(1,3),np.dtype[np.floating]]]
# Array2 = np.ndarray[Any, np.dtype[np.floating]]
# Array3 = np.ndarray[Any, np.dtype[np.floating]]
Array6 = ndarray[Tuple[L[6,]], dtype[floating]]

QuaternionArray = ndarray[Tuple[L[4,]], dtype[floating]]
UnitQuaternionArray = ndarray[Tuple[L[4,]], dtype[floating]]
QuaternionArrayx = Union[List,Tuple[float,float,float,float],ndarray[Tuple[L[4,]], dtype[floating]]]
UnitQuaternionArrayx = Union[List,Tuple[float,float,float,float],ndarray[Tuple[L[4,]], dtype[floating]]]

# R2x = Union[List[float],Tuple[float,float],Array2]  # various ways to represent R^3 for input
# R3x = Union[List[float],Tuple[float,float],Array3]  # various ways to represent R^3 for input
R6x = Union[List[float],Tuple[float,float,float,float,float,float],Array6]  # various ways to represent R^3 for input

# R2 = np.ndarray[Any, np.dtype[np.floating]]  # R^2
# R3 = np.ndarray[Any, np.dtype[np.floating]]  # R^3
# R6 = np.ndarray[Any, np.dtype[np.floating]]  # R^6
# SO2 = np.ndarray[Any, np.dtype[np.floating]]  # SO(3) rotation matrix
# SE2 = np.ndarray[Any, np.dtype[np.floating]]  # SE(3) rigid-body transform
# SO3 = np.ndarray[Any, np.dtype[np.floating]]  # SO(3) rotation matrix
# SE3 = np.ndarray[Any, np.dtype[np.floating]]  # SE(3) rigid-body transform
SOnArray = Union[SO2Array,SO3Array]
SEnArray = Union[SE2Array,SE3Array]

so2 = ndarray[Tuple[L[3,3]], dtype[floating]]  # so(2) Lie algebra of SO(2), skew-symmetrix matrix
se2 = ndarray[Tuple[L[3,3]], dtype[floating]]  # se(2) Lie algebra of SE(2), augmented skew-symmetrix matrix
so3 = ndarray[Tuple[L[3,3]], dtype[floating]]  # so(3) Lie algebra of SO(3), skew-symmetrix matrix
se3 = ndarray[Tuple[L[3,3]], dtype[floating]]  # se(3) Lie algebra of SE(3), augmented skew-symmetrix matrix
sonArray = Union[so2Array,so3Array]
senArray = Union[se2Array,se3Array]

Rn = Union[R2,R3]
