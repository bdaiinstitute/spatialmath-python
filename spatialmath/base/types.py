import sys

_version = sys.version_info.minor

# from spatialmath.base._types_39 import *

if _version >= 11:
    from spatialmath.base._types_311 import *
elif _version >= 9:
    from spatialmath.base._types_311 import *
else:
    from spatialmath.base._types_311 import *

# pass
