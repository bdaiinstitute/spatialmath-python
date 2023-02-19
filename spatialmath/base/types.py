import sys

_version = sys.version_info.minor


if _version >= 11:
    from spatialmath.base._types_311 import *
elif _version >= 9:
    from spatialmath.base._types_39 import *
else:
    from spatialmath.base._types_35 import *
