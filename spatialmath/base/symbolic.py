import math

try:  # pragma: no cover
    # print('Using SymPy')
    import sympy
    from sympy import S

    _symbolics = True
    symtype = (sympy.Expr,)

except ImportError:
    _symbolics = False
    symtype = ()


# ---------------------------------------------------------------------------------------#

def symbol(name, real=True):
    return sympy.symbols(name, real=real)

def issymbol(var):
    if _symbolics:
        if isinstance(var, (list, tuple)):
            return any([isinstance(x, symtype) for x in var])
        else:
            return isinstance(var, symtype)
    else:
        return False

def sin(theta):
    if issymbol(theta):
        return sympy.sin(theta)
    else:
        return math.sin(theta)

def cos(theta):
    if issymbol(theta):
        return sympy.cos(theta)
    else:
        return math.cos(theta)

def sqrt(v):
    if issymbol(v):
        return sympy.sqrt(v)
    else:
        return math.sqrt(v)    

def zero():
    return S.Zero

def one():
    return S.One

def negative_one():
    return S.NegativeOne

def zero():
    return S.Zero

def pi():
    return S.Pi

def simplify(x):
    if _symbolics:
        return sympy.simplify(x)
    else:
        return x

