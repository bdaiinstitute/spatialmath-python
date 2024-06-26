import sys

# For type checking to work, we have to use the complete version_info,
# we can't use the .minor_version attribute

if sys.version_info >= (3, 11):
    from spatialmath.base._types_311 import (
        ArrayLike,
        ArrayLikePure,
        ArrayLike2,
        ArrayLike3,
        ArrayLike4,
        ArrayLike6,
        R1,
        R2,
        R3,
        R4,
        R6,
        R8,
        R1x1,
        R2x2,
        R3x3,
        R4x4,
        R6x6,
        R8x8,
        R1x3,
        R3x1,
        R1x2,
        R2x1,
        Points2,
        Points3,
        RNx3,
        SO2Array,
        SE2Array,
        SO3Array,
        SE3Array,
        so2Array,
        se2Array,
        so3Array,
        se3Array,
        QuaternionArray,
        UnitQuaternionArray,
        Rn,
        SOnArray,
        SEnArray,
        sonArray,
        senArray,
        Color,
    )
elif sys.version_info >= (3, 9):
    from spatialmath.base._types_39 import (
        ArrayLike,
        ArrayLikePure,
        ArrayLike2,
        ArrayLike3,
        ArrayLike4,
        ArrayLike6,
        R1,
        R2,
        R3,
        R4,
        R6,
        R8,
        R1x1,
        R2x2,
        R3x3,
        R4x4,
        R6x6,
        R8x8,
        R1x3,
        R3x1,
        R1x2,
        R2x1,
        Points2,
        Points3,
        RNx3,
        SO2Array,
        SE2Array,
        SO3Array,
        SE3Array,
        so2Array,
        se2Array,
        so3Array,
        se3Array,
        QuaternionArray,
        UnitQuaternionArray,
        Rn,
        SOnArray,
        SEnArray,
        sonArray,
        senArray,
        Color,
    )
else:
    from spatialmath.base._types_35 import (
        ArrayLike,
        ArrayLikePure,
        ArrayLike2,
        ArrayLike3,
        ArrayLike4,
        ArrayLike6,
        R1,
        R2,
        R3,
        R4,
        R6,
        R8,
        R1x1,
        R2x2,
        R3x3,
        R4x4,
        R6x6,
        R8x8,
        R1x3,
        R3x1,
        R1x2,
        R2x1,
        Points2,
        Points3,
        RNx3,
        SO2Array,
        SE2Array,
        SO3Array,
        SE3Array,
        so2Array,
        se2Array,
        so3Array,
        se3Array,
        QuaternionArray,
        UnitQuaternionArray,
        Rn,
        SOnArray,
        SEnArray,
        sonArray,
        senArray,
        Color,
    )

ArrayLikePure = ArrayLikePure
ArrayLike = ArrayLike
ArrayLike2 = ArrayLike2
ArrayLike3 = ArrayLike3
ArrayLike4 = ArrayLike4
ArrayLike6 = ArrayLike6
R1 = R1
R2 = R2
R3 = R3
R4 = R4
R6 = R6
R8 = R8
R1x1 = R1x1
R2x2 = R2x2
R3x3 = R3x3
R4x4 = R4x4
R6x6 = R6x6
R8x8 = R8x8
R1x3 = R1x3
R3x1 = R3x1
R1x2 = R1x2
R2x1 = R2x1
Points2 = Points2
Points3 = Points3
RNx3 = RNx3
SO2Array = SO2Array
SE2Array = SE2Array
SO3Array = SO3Array
SE3Array = SE3Array
so2Array = so2Array
se2Array = se2Array
so3Array = so3Array
se3Array = se3Array
QuaternionArray = QuaternionArray
UnitQuaternionArray = UnitQuaternionArray
Rn = Rn
SOnArray = SOnArray
SEnArray = SEnArray
sonArray = sonArray
senArray = senArray
Color = Color
