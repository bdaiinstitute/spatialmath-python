""" 
This modules contains functions to create and transform 3D rotation matrices
and homogeneous tranformation matrices.

Vector arguments are what numpy refers to as ``array_like`` and can be a list,
tuple, numpy array, numpy row vector or numpy column vector.

Versions: 
    
    1. Luis Fernando Lara Tobar and Peter Corke, 2008
    2. Josh Carrigg Hodson, Aditya Dua, Chee Ho Chan, 2017
    3. Peter Corke, 2020
    
TODO:
        
    - trinterp
    - trjac, trjac2
    - tranimate, tranimate2
"""

import sys
import math
import numpy as np
from spatialmath.base import argcheck
from spatialmath.base import vectors as vec
from spatialmath.base import transformsNd as trn


try: # pragma: no cover
    # print('Using SymPy')
    import sympy as sym
    def issymbol(x):
        return isinstance(x, sym.Symbol)
except:
    def issymbol(x):
        return False
    
_eps = np.finfo(np.float64).eps

def colvec(v):
    return np.array(v).reshape((len(v), 1))

# ---------------------------------------------------------------------------------------#
    
def _cos(theta):
    if issymbol(theta):
        return sym.cos(theta)
    else:
        return math.cos(theta)
        
def _sin(theta):
    if issymbol(theta):
        return sym.sin(theta)
    else:
        return math.sin(theta)

def rotx(theta, unit="rad"):
    """
    Create SO(3) rotation about X-axis

    :param theta: rotation angle about X-axis
    :type theta: float
    :param unit: angular units: 'rad' [default], or 'deg'
    :type unit: str
    :return: 3x3 rotation matrix
    :rtype: numpy.ndarray, shape=(3,3)

    - ``rotx(THETA)`` is an SO(3) rotation matrix (3x3) representing a rotation
      of THETA radians about the x-axis
    - ``rotx(THETA, "deg")`` as above but THETA is in degrees
    
    :seealso: :func:`~trotx`
    """

    theta = argcheck.getunit(theta, unit)
    ct = _cos(theta)
    st = _sin(theta)
    R = np.array([
            [1, 0, 0], 
            [0, ct, -st], 
            [0, st, ct]  ])
    return R


# ---------------------------------------------------------------------------------------#
def roty(theta, unit="rad"):
    """
    Create SO(3) rotation about Y-axis

    :param theta: rotation angle about Y-axis
    :type theta: float
    :param unit: angular units: 'rad' [default], or 'deg'
    :type unit: str
    :return: 3x3 rotation matrix
    :rtype: numpy.ndarray, shape=(3,3)

    - ``roty(THETA)`` is an SO(3) rotation matrix (3x3) representing a rotation
      of THETA radians about the y-axis
    - ``roty(THETA, "deg")`` as above but THETA is in degrees
    
    :seealso: :func:`~troty`
    """

    theta = argcheck.getunit(theta, unit)
    ct = _cos(theta)
    st = _sin(theta)
    R = np.array([
            [ct, 0, st], 
            [0, 1, 0], 
            [-st, 0, ct]  ])
    return R


# ---------------------------------------------------------------------------------------#
def rotz(theta, unit="rad"):
    """
    Create SO(3) rotation about Z-axis

    :param theta: rotation angle about Z-axis
    :type theta: float
    :param unit: angular units: 'rad' [default], or 'deg'
    :type unit: str
    :return: 3x3 rotation matrix
    :rtype: numpy.ndarray, shape=(3,3)

    - ``rotz(THETA)`` is an SO(3) rotation matrix (3x3) representing a rotation
      of THETA radians about the z-axis
    - ``rotz(THETA, "deg")`` as above but THETA is in degrees
    
    :seealso: :func:`~yrotz`
    """
    theta = argcheck.getunit(theta, unit)
    ct = _cos(theta)
    st = _sin(theta)
    R = np.array([
            [ct, -st, 0], 
            [st,  ct, 0], 
            [0,   0,  1]  ])
    return R


# ---------------------------------------------------------------------------------------#
def trotx(theta, unit="rad", t=None):
    """
    Create SE(3) pure rotation about X-axis

    :param theta: rotation angle about X-axis
    :type theta: float
    :param unit: angular units: 'rad' [default], or 'deg'
    :type unit: str
    :param t: translation 3-vector, defaults to [0,0,0]
    :type t: array_like    :return: 4x4 homogeneous transformation matrix
    :rtype: numpy.ndarray, shape=(4,4)

    - ``trotx(THETA)`` is a homogeneous transformation (4x4) representing a rotation
      of THETA radians about the x-axis.
    - ``trotx(THETA, 'deg')`` as above but THETA is in degrees
    - ``trotx(THETA, 'rad', t=[x,y,z])`` as above with translation of [x,y,z]
    
    :seealso: :func:`~rotx`
    """
    T  = np.pad( rotx(theta, unit), (0,1), mode='constant' )
    if t is not None:
        T[:3,3] = argcheck.getvector(t, 3, 'array')
    T[3,3] = 1.0
    return T


# ---------------------------------------------------------------------------------------#
def troty(theta, unit="rad", t=None):
    """
    Create SE(3) pure rotation about Y-axis

    :param theta: rotation angle about Y-axis
    :type theta: float
    :param unit: angular units: 'rad' [default], or 'deg'
    :type unit: str
    :param t: translation 3-vector, defaults to [0,0,0]
    :type t: array_like
    :return: 4x4 homogeneous transformation matrix as a numpy array
    :rtype: numpy.ndarray, shape=(4,4)

    - ``troty(THETA)`` is a homogeneous transformation (4x4) representing a rotation
      of THETA radians about the y-axis.
    - ``troty(THETA, 'deg')`` as above but THETA is in degrees
    - ``troty(THETA, 'rad', t=[x,y,z])`` as above with translation of [x,y,z]
    
    :seealso: :func:`~roty`
    """
    T  = np.pad( roty(theta, unit), (0,1), mode='constant' )
    if t is not None:
        T[:3,3] = argcheck.getvector(t, 3, 'array')
    T[3,3] = 1.0
    return T


# ---------------------------------------------------------------------------------------#
def trotz(theta, unit="rad", t=None):
    """
    Create SE(3) pure rotation about Z-axis

    :param theta: rotation angle about Z-axis
    :type theta: float
    :param unit: angular units: 'rad' [default], or 'deg'
    :type unit: str
    :param t: translation 3-vector, defaults to [0,0,0]
    :type t: array_like
    :return: 4x4 homogeneous transformation matrix
    :rtype: numpy.ndarray, shape=(4,4)

    - ``trotz(THETA)`` is a homogeneous transformation (4x4) representing a rotation
      of THETA radians about the z-axis.
    - ``trotz(THETA, 'deg')`` as above but THETA is in degrees
    - ``trotz(THETA, 'rad', t=[x,y,z])`` as above with translation of [x,y,z]
    
    :seealso: :func:`~rotz`
    """
    T  = np.pad( rotz(theta, unit), (0,1), mode='constant' )
    if t is not None:
        T[:3,3] = argcheck.getvector(t, 3, 'array')
    T[3,3] = 1.0
    return T

# ---------------------------------------------------------------------------------------#
def transl(x, y=None, z=None):
    """
    Create SE(3) pure translation, or extract translation from SE(3) matrix

    :param x: translation along X-axis
    :type x: float
    :param y: translation along Y-axis
    :type y: float
    :param z: translation along Z-axis
    :type z: float
    :return: 4x4 homogeneous transformation matrix
    :rtype: numpy.ndarray, shape=(4,4)

    Create a translational SE(3) matrix:

    - ``T = transl( X, Y, Z )`` is an SE(3) homogeneous transform (4x4) representing a
      pure translation of X, Y and Z.
    - ``T = transl( V )`` as above but the translation is given by a 3-element
      list, dict, or a numpy array, row or column vector.


    Extract the translational part of an SE(3) matrix:

    - ``P = TRANSL(T)`` is the translational part of a homogeneous transform T as a
      3-element numpy array.
    
    :seealso: :func:`~spatialmath.base.transforms2d.transl2`
   """

    if np.isscalar(x):
        T = np.identity(4)
        T[:3,3] = [x, y, z]
        return T
    elif argcheck.isvector(x, 3):
        T = np.identity(4)
        T[:3,3] = argcheck.getvector(x, 3, out='array')
        return T
    elif argcheck.ismatrix(x, (4,4)):
        return x[:3,3]
    else:
        ValueError('bad argument')
        


def ishom(T, check=False, tol=10):
    """
    Test if matrix belongs to SE(3)
    
    :param T: matrix to test
    :type T: numpy.ndarray
    :param check: check validity of rotation submatrix
    :type check: bool
    :return: whether matrix is an SE(3) homogeneous transformation matrix
    :rtype: bool
    
    - ``ISHOM(T)`` is True if the argument ``T`` is of dimension 4x4
    - ``ISHOM(T, check=True)`` as above, but also checks orthogonality of the rotation sub-matrix and 
      validitity of the bottom row.
    
    :seealso: :func:`~spatialmath.base.transformsNd.isR`, :func:`~isrot`, :func:`~spatialmath.base.transforms2d.ishom2`
    """
    return T.shape == (4,4) and (not check or (trn.isR(T[:3,:3], tol=tol) and np.all(T[3,:] == np.array([0,0,0,1]))))

def isrot(R, check=False, tol=10):
    """
    Test if matrix belongs to SO(3)
    
    :param R: matrix to test
    :type R: numpy.ndarray
    :param check: check validity of rotation submatrix
    :type check: bool
    :return: whether matrix is an SO(3) rotation matrix
    :rtype: bool
    
    - ``ISROT(R)`` is True if the argument ``R`` is of dimension 3x3
    - ``ISROT(R, check=True)`` as above, but also checks orthogonality of the rotation matrix.
    
    :seealso: :func:`~spatialmath.base.transformsNd.isR`, :func:`~spatialmath.base.transforms2d.isrot2`,  :func:`~ishom`
    """
    return R.shape == (3,3) and (not check or trn.isR(R, tol=tol))


# ---------------------------------------------------------------------------------------#
def rpy2r(roll, pitch=None, yaw=None, *, unit='rad', order='zyx'):
    """
    Create an SO(3) rotation matrix from roll-pitch-yaw angles

    :param roll: roll angle
    :type roll: float
    :param pitch: pitch angle
    :type pitch: float
    :param yaw: yaw angle
    :type yaw: float
    :param unit: angular units: 'rad' [default], or 'deg'
    :type unit: str
    :param unit: rotation order: 'zyx' [default], 'xyz', or 'yxz'
    :type unit: str
    :return: 3x3 rotation matrix
    :rtype: numpdy.ndarray, shape=(3,3)

    - ``rpy2r(ROLL, PITCH, YAW)`` is an SO(3) orthonormal rotation matrix
      (3x3) equivalent to the specified roll, pitch, yaw angles angles.
      These correspond to successive rotations about the axes specified by ``order``:
          
        - 'zyx' [default], rotate by yaw about the z-axis, then by pitch about the new y-axis,
          then by roll about the new x-axis.  Convention for a mobile robot with x-axis forward
          and y-axis sideways.
        - 'xyz', rotate by yaw about the x-axis, then by pitch about the new y-axis,
          then by roll about the new z-axis. Covention for a robot gripper with z-axis forward
          and y-axis between the gripper fingers.
        - 'yxz', rotate by yaw about the y-axis, then by pitch about the new x-axis,
          then by roll about the new z-axis. Convention for a camera with z-axis parallel
          to the optic axis and x-axis parallel to the pixel rows.
          
    - ``rpy2r(RPY)`` as above but the roll, pitch, yaw angles are taken
      from ``RPY`` which is a 3-vector (array_like) with values
      (ROLL, PITCH, YAW).
      
    :seealso: :func:`~eul2r`, :func:`~rpy2tr`, :func:`~tr2rpy`
    """
    
    if np.isscalar(roll):
        angles = [roll, pitch, yaw]
    else:
        angles = argcheck.getvector(roll, 3)
        
    angles = argcheck.getunit(angles, unit)
    
    if order == 'xyz' or order == 'arm':
        R = rotx(angles[2]) @ roty(angles[1]) @ rotz(angles[0])
    elif order == 'zyx' or order == 'vehicle':
        R = rotz(angles[2]) @ roty(angles[1]) @ rotx(angles[0])
    elif order == 'yxz' or order == 'camera':
        R = roty(angles[2]) @ rotx(angles[1]) @ rotz(angles[0])
    else:
        raise ValueError('Invalid angle order')
        
    return R


# ---------------------------------------------------------------------------------------#
def rpy2tr(roll, pitch=None, yaw=None, unit='rad', order='zyx'):
    """
    Create an SE(3) rotation matrix from roll-pitch-yaw angles

    :param roll: roll angle
    :type roll: float
    :param pitch: pitch angle
    :type pitch: float
    :param yaw: yaw angle
    :type yaw: float
    :param unit: angular units: 'rad' [default], or 'deg'
    :type unit: str
    :param unit: rotation order: 'zyx' [default], 'xyz', or 'yxz'
    :type unit: str
    :return: 3x3 rotation matrix
    :rtype: numpdy.ndarray, shape=(3,3)

    - ``rpy2tr(ROLL, PITCH, YAW)`` is an SO(3) orthonormal rotation matrix
      (3x3) equivalent to the specified roll, pitch, yaw angles angles.
      These correspond to successive rotations about the axes specified by ``order``:
          
        - 'zyx' [default], rotate by yaw about the z-axis, then by pitch about the new y-axis,
          then by roll about the new x-axis.  Convention for a mobile robot with x-axis forward
          and y-axis sideways.
        - 'xyz', rotate by yaw about the x-axis, then by pitch about the new y-axis,
          then by roll about the new z-axis. Convention for a robot gripper with z-axis forward
          and y-axis between the gripper fingers.
        - 'yxz', rotate by yaw about the y-axis, then by pitch about the new x-axis,
          then by roll about the new z-axis. Convention for a camera with z-axis parallel
          to the optic axis and x-axis parallel to the pixel rows.
          
    - ``rpy2tr(RPY)`` as above but the roll, pitch, yaw angles are taken
      from ``RPY`` which is a 3-vector (array_like) with values
      (ROLL, PITCH, YAW). 
    
    Notes:
        
    - The translational part is zero.
    
    :seealso: :func:`~eul2tr`, :func:`~rpy2r`, :func:`~tr2rpy`
    """

    R = rpy2r(roll, pitch, yaw, order=order, unit=unit)
    return trn.r2t(R)

# ---------------------------------------------------------------------------------------#
def eul2r(phi, theta=None, psi=None, unit='rad'):
    """
    Create an SO(3) rotation matrix from Euler angles

    :param phi: Z-axis rotation
    :type phi: float
    :param theta: Y-axis rotation
    :type theta: float
    :param psi: Z-axis rotation
    :type psi: float
    :param unit: angular units: 'rad' [default], or 'deg'
    :type unit: str
    :return: 3x3 rotation matrix
    :rtype: numpdy.ndarray, shape=(3,3)

    - ``R = eul2r(PHI, THETA, PSI)`` is an SO(3) orthonornal rotation
      matrix equivalent to the specified Euler angles.  These correspond
      to rotations about the Z, Y, Z axes respectively.
    - ``R = eul2r(EUL)`` as above but the Euler angles are taken from
      ``EUL`` which is a 3-vector (array_like) with values
      (PHI THETA PSI).
      
    :seealso: :func:`~rpy2r`, :func:`~eul2tr`, :func:`~tr2eul`
    """
    
    if np.isscalar(phi):
        angles = [phi, theta, psi]
    else:
        angles = argcheck.getvector(phi, 3)
        
    angles = argcheck.getunit(angles, unit)
        
    return rotz(angles[0]) @ roty(angles[1]) @ rotz(angles[2])


# ---------------------------------------------------------------------------------------#
def eul2tr(phi, theta=None, psi=None, unit='rad'):
    """
    Create an SE(3) pure rotation matrix from Euler angles

    :param phi: Z-axis rotation
    :type phi: float
    :param theta: Y-axis rotation
    :type theta: float
    :param psi: Z-axis rotation
    :type psi: float
    :param unit: angular units: 'rad' [default], or 'deg'
    :type unit: str
    :return: 4x4 homogeneous transformation matrix
    :rtype: numpdy.ndarray, shape=(4,4)

    - ``R = eul2tr(PHI, THETA, PSI)`` is an SE(3) homogeneous transformation
      matrix equivalent to the specified Euler angles.  These correspond
      to rotations about the Z, Y, Z axes respectively.
    - ``R = eul2tr(EUL)`` as above but the Euler angles are taken from 
      ``EUL`` which is a 3-vector (array_like) with values
      (PHI THETA PSI). 
    
    Notes:
        
    - The translational part is zero.

    :seealso: :func:`~rpy2tr`, :func:`~eul2r`, :func:`~tr2eul`
    """
    
    R = eul2r(phi, theta, psi, unit=unit)
    return trn.r2t(R)

# ---------------------------------------------------------------------------------------#
def angvec2r(theta, v, unit='rad'):
    """
    Create an SO(3) rotation matrix from rotation angle and axis

    :param theta: rotation
    :type theta: float
    :param unit: angular units: 'rad' [default], or 'deg'
    :type unit: str
    :param v: rotation axis, 3-vector
    :type v: array_like
    :return: 3x3 rotation matrix
    :rtype: numpdy.ndarray, shape=(3,3)
    
    ``angvec2r(THETA, V)`` is an SO(3) orthonormal rotation matrix
    equivalent to a rotation of ``THETA`` about the vector ``V``.
    
    Notes:
        
    - If ``THETA == 0`` then return identity matrix.
    - If ``THETA ~= 0`` then ``V`` must have a finite length.

    :seealso: :func:`~angvec2tr`, :func:`~tr2angvec`
    """
    assert np.isscalar(theta) and argcheck.isvector(v, 3), "Arguments must be theta and vector"
    
    if np.linalg.norm(v) < 10*_eps:
            return np.eye(3)
        
    theta = argcheck.getunit(theta, unit)
    
    # Rodrigue's equation

    sk = trn.skew( vec.unitvec(v) )
    R = np.eye(3) + math.sin(theta) * sk + (1.0 - math.cos(theta)) * sk @ sk
    return R


# ---------------------------------------------------------------------------------------#
def angvec2tr(theta, v, unit='rad'):
    """
    Create an SE(3) pure rotation from rotation angle and axis

    :param theta: rotation
    :type theta: float
    :param unit: angular units: 'rad' [default], or 'deg'
    :type unit: str
    :param v: rotation axis, 3-vector
    :type v: : array_like
    :return: 4x4 homogeneous transformation matrix
    :rtype: numpdy.ndarray, shape=(4,4)
    
    ``angvec2tr(THETA, V)`` is an SE(3) homogeneous transformation matrix
    equivalent to a rotation of ``THETA`` about the vector ``V``.
    
    Notes:
        
    - If ``THETA == 0`` then return identity matrix.
    - If ``THETA ~= 0`` then ``V`` must have a finite length.
    - The translational part is zero.

    :seealso: :func:`~angvec2r`, :func:`~tr2angvec`
    """
    return trn.r2t(angvec2r(theta, v, unit=unit))


# ---------------------------------------------------------------------------------------#
def oa2r(o, a=None):
    """
    Create SO(3) rotation matrix from two vectors

    :param o: 3-vector parallel to Y- axis
    :type o: array_like
    :param a: 3-vector parallel to the Z-axis
    :type o: array_like
    :return: 3x3 rotation matrix
    :rtype: numpy.ndarray, shape=(3,3)

    ``T = oa2tr(O, A)`` is an SO(3) orthonormal rotation matrix for a frame defined in terms of
    vectors parallel to its Y- and Z-axes with respect to a reference frame.  In robotics these axes are 
    respectively called the orientation and approach vectors defined such that
    R = [N O A] and N = O x A.
    
    Steps:
        
        1. N' = O x A
        2. O' = A x N
        3. normalize N', O', A
        4. stack horizontally into rotation matrix

    Notes:
        
    - The A vector is the only guaranteed to have the same direction in the resulting 
      rotation matrix
    - O and A do not have to be unit-length, they are normalized
    - O and A do not have to be orthogonal, so long as they are not parallel
    - The vectors O and A are parallel to the Y- and Z-axes of the equivalent coordinate frame.

    :seealso: :func:`~oa2tr`
    """
    o = argcheck.getvector(o, 3, out='array')
    a = argcheck.getvector(a, 3, out='array')
    n = np.cross(o, a)
    o = np.cross(a, n)
    R = np.stack( (vec.unitvec(n), vec.unitvec(o), vec.unitvec(a)), axis=1)
    return R


# ---------------------------------------------------------------------------------------#
def oa2tr(o, a=None):
    """
    Create SE(3) pure rotation from two vectors

    :param o: 3-vector parallel to Y- axis
    :type o: array_like
    :param a: 3-vector parallel to the Z-axis
    :type o: array_like
    :return: 4x4 homogeneous transformation matrix
    :rtype: numpy.ndarray, shape=(4,4)

    ``T = oa2tr(O, A)`` is an SE(3) homogeneous transformation matrix for a frame defined in terms of
    vectors parallel to its Y- and Z-axes with respect to a reference frame.  In robotics these axes are 
    respectively called the orientation and approach vectors defined such that
    R = [N O A] and N = O x A.
    
    Steps:
        
        1. N' = O x A
        2. O' = A x N
        3. normalize N', O', A
        4. stack horizontally into rotation matrix

    Notes:
        
    - The A vector is the only guaranteed to have the same direction in the resulting 
      rotation matrix
    - O and A do not have to be unit-length, they are normalized
    - O and A do not have to be orthogonal, so long as they are not parallel
    - The translational part is zero.
    - The vectors O and A are parallel to the Y- and Z-axes of the equivalent coordinate frame.

    :seealso: :func:`~oa2r`
    """
    return trn.r2t(oa2r(o, a))


# ------------------------------------------------------------------------------------------------------------------- #
def tr2angvec(T, unit='rad', check=False):
    r"""
    Convert SO(3) or SE(3) to angle and rotation vector
    
    :param R: SO(3) or SE(3) matrix
    :type R: numpy.ndarray, shape=(3,3) or (4,4)
    :param unit: 'rad' or 'deg'
    :type unit: str
    :param check: check that rotation matrix is valid
    :type check: bool
    :return: :math:`(\theta, {\bf v})`
    :rtype: float, numpy.ndarray, shape=(3,)
    
    ``tr2angvec(R)`` is a rotation angle and a vector about which the rotation
    acts that corresponds to the rotation part of ``R``. 
        
    By default the angle is in radians but can be changed setting `unit='deg'`.
    
    Notes:
        
    - If the input is SE(3) the translation component is ignored.
    
    :seealso: :func:`~angvec2r`, :func:`~angvec2tr`, :func:`~tr2rpy`, :func:`~tr2eul`
    """

    if argcheck.ismatrix(T, (4,4)):
        R = trn.t2r(T)
    else:
        R = T
    assert isrot(R, check=check)
    
    v = trn.vex( trlog(R) )
    
    if vec.iszerovec(v):
        theta = 0
        v = np.r_[0, 0, 0]
    else:
        theta = vec.norm(v)
        v = vec.unitvec(v)
    
    if unit == 'deg':
        theta *= 180 / math.pi
               
    return (theta, v)


# ------------------------------------------------------------------------------------------------------------------- #
def tr2eul(T, unit='rad', flip=False, check=False):
    r"""
    Convert SO(3) or SE(3) to ZYX Euler angles
    
    :param R: SO(3) or SE(3) matrix
    :type R: numpy.ndarray, shape=(3,3) or (4,4)
    :param unit: 'rad' or 'deg'
    :type unit: str
    :param flip: choose first Euler angle to be in quadrant 2 or 3
    :type flip: bool
    :param check: check that rotation matrix is valid
    :type check: bool
    :return: ZYZ Euler angles
    :rtype: numpy.ndarray, shape=(3,)
    
    ``tr2eul(R)`` are the Euler angles corresponding to 
    the rotation part of ``R``. 
    
    The 3 angles :math:`[\phi, \theta, \psi` correspond to sequential rotations about the
    Z, Y and Z axes respectively.
    
    By default the angles are in radians but can be changed setting `unit='deg'`.
    
    Notes:
        
    - There is a singularity for the case where :math:`\theta=0` in which case :math:`\phi` is arbitrarily set to zero and :math:`\phi` is set to :math:`\phi+\psi`.
    - If the input is SE(3) the translation component is ignored.
    
    :seealso: :func:`~eul2r`, :func:`~eul2tr`, :func:`~tr2rpy`, :func:`~tr2angvec`
    """
    
    if argcheck.ismatrix(T, (4,4)):
        R = trn.t2r(T)
    else:
        R = T
    assert isrot(R, check=check)
    
    eul = np.zeros((3,))
    if abs(R[0,2]) < 10*_eps and abs(R[1,2]) < 10*_eps:
        eul[0] = 0
        sp = 0
        cp = 1
        eul[1] = math.atan2(cp * R[0,2] + sp * R[1,2],R[2,2])
        eul[2] = math.atan2(-sp * R[0,0] + cp * R[1,0], -sp * R[0,1] + cp * R[1,1])
    else:
        if flip:
            eul[0] = math.atan2(-R[1,2], -R[0,2])
        else:
            eul[0] = math.atan2(R[1,2],R[0,2])
        sp = math.sin(eul[0])
        cp = math.cos(eul[0])
        eul[1] = math.atan2(cp * R[0,2] + sp * R[1,2], R[2,2])
        eul[2] = math.atan2(-sp * R[0,0] + cp * R[1,0], -sp * R[0,1] + cp * R[1, 1])

    if unit == 'deg':
        eul *= 180 / math.pi

    return eul

# ------------------------------------------------------------------------------------------------------------------- #
def tr2rpy(T, unit='rad', order='zyx', check=False):
    """
    Convert SO(3) or SE(3) to roll-pitch-yaw angles
    
    :param R: SO(3) or SE(3) matrix
    :type R: numpy.ndarray, shape=(3,3) or (4,4)
    :param unit: 'rad' or 'deg'
    :type unit: str
    :param order: 'xyz', 'zyx' or 'yxz' [default 'zyx']
    :type unit: str
    :param check: check that rotation matrix is valid
    :type check: bool
    :return: Roll-pitch-yaw angles
    :rtype: numpy.ndarray, shape=(3,)
    
    ``tr2rpy(R)`` are the roll-pitch-yaw angles corresponding to 
    the rotation part of ``R``. 
    
    The 3 angles RPY=[R,P,Y] correspond to sequential rotations about the
    Z, Y and X axes respectively.  The axis order sequence can be changed by
    setting:
 
    - `order='xyz'`  for sequential rotations about X, Y, Z axes
    - `order='yxz'`  for sequential rotations about Y, X, Z axes
    
    By default the angles are in radians but can be changed setting `unit='deg'`.
    
    Notes:
        
    - There is a singularity for the case where P=:math:`\pi/2` in which case R is arbitrarily set to zero and Y is the sum (R+Y).
    - If the input is SE(3) the translation component is ignored.
    
    :seealso: :func:`~rpy2r`, :func:`~rpy2tr`, :func:`~tr2eul`, :func:`~tr2angvec`
    """
    
    if argcheck.ismatrix(T, (4,4)):
        R = trn.t2r(T)
    else:
        R = T
    assert isrot(R, check=check)
    
    rpy = np.zeros((3,))
    if order == 'xyz' or order == 'arm':

        # XYZ order
        if abs(abs(R[0,2]) - 1) < 10*_eps:  # when |R13| == 1
            # singularity
            rpy[0] = 0  # roll is zero
            if R[0,2] > 0:
                rpy[2] = math.atan2( R[2,1], R[1,1])   # R+Y
            else:
                rpy[2] = -math.atan2( R[1,0], R[2,0])   # R-Y
            rpy[1] = math.asin(R[0,2])
        else:
            rpy[0] = -math.atan2(R[0,1], R[0,0])
            rpy[2] = -math.atan2(R[1,2], R[2,2])
            
            k = np.argmax(np.abs( [R[0,0], R[0,1], R[1,2], R[2,2]] ))
            if k == 0:
                rpy[1] =  math.atan(R[0,2]*math.cos(rpy[0])/R[0,0])
            elif k == 1:
                rpy[1] = -math.atan(R[0,2]*math.sin(rpy[0])/R[0,1])
            elif k == 2:
                rpy[1] = -math.atan(R[0,2]*math.sin(rpy[2])/R[1,2])
            elif k == 3:
                rpy[1] =  math.atan(R[0,2]*math.cos(rpy[2])/R[2,2])
                                        

    elif order == 'zyx' or order == 'vehicle':

        # old ZYX order (as per Paul book)
        if abs(abs(R[2,0]) - 1) < 10*_eps:  # when |R31| == 1
            # singularity
            rpy[0] = 0     # roll is zero
            if R[2,0] < 0:
                rpy[2] = -math.atan2(R[0,1], R[0,2])  # R-Y
            else:
                rpy[2] = math.atan2(-R[0,1], -R[0,2])  # R+Y
            rpy[1] = -math.asin(R[2,0])
        else:
            rpy[0] = math.atan2(R[2,1], R[2,2])  # R
            rpy[2] = math.atan2(R[1,0], R[0,0])  # Y
                 
            k = np.argmax(np.abs( [R[0,0], R[1,0], R[2,1], R[2,2]] ))
            if k == 0:
                rpy[1] = -math.atan(R[2,0]*math.cos(rpy[2])/R[0,0])
            elif k == 1:
                rpy[1] = -math.atan(R[2,0]*math.sin(rpy[2])/R[1,0])
            elif k == 2:
                rpy[1] = -math.atan(R[2,0]*math.sin(rpy[0])/R[2,1])
            elif k == 3:
                rpy[1] = -math.atan(R[2,0]*math.cos(rpy[0])/R[2,2])

    elif order == 'yxz' or order == 'camera':

            if abs(abs(R[1,2]) - 1) < 10*_eps:  # when |R23| == 1
                # singularity
                rpy[0] = 0
                if R[1,2] < 0:
                    rpy[2] = -math.atan2(R[2,0], R[0,0])   # R-Y
                else:
                    rpy[2] = math.atan2(-R[2,0], -R[2,1])   # R+Y
                rpy[1] = -math.asin(R[1,2])    # P
            else:
                rpy[0] = math.atan2(R[1,0], R[1,1])
                rpy[2] = math.atan2(R[0,2], R[2,2])
                
                k = np.argmax(np.abs( [R[1,0], R[1,1], R[0,2], R[2,2]] ))
                if k == 0:
                    rpy[1] = -math.atan(R[1,2]*math.sin(rpy[0])/R[1,0])
                elif k == 1:
                    rpy[1] = -math.atan(R[1,2]*math.cos(rpy[0])/R[1,1])
                elif k == 2:
                    rpy[1] = -math.atan(R[1,2]*math.sin(rpy[2])/R[0,2])
                elif k == 3:
                    rpy[1] = -math.atan(R[1,2]*math.cos(rpy[2])/R[2,2])  

    else:
        raise ValueError('Invalid order')

    if unit == 'deg':
        rpy *= 180 / math.pi

    return rpy


# ---------------------------------------------------------------------------------------#
def trlog(T, check=True):
    """
    Logarithm of SO(3) or SE(3) matrix

    :param T: SO(3) or SE(3) matrix
    :type T: numpy.ndarray, shape=(3,3) or (4,4)
    :return: logarithm
    :rtype: numpy.ndarray, shape=(3,3) or (4,4)
    :raises: ValueError

    An efficient closed-form solution of the matrix logarithm for arguments that are SO(3) or SE(3).
    
    - ``trlog(R)`` is the logarithm of the passed rotation matrix ``R`` which will be 
      3x3 skew-symmetric matrix.  The equivalent vector from ``vex()`` is parallel to rotation axis
      and its norm is the amount of rotation about that axis.
    - ``trlog(T)`` is the logarithm of the passed homogeneous transformation matrix ``T`` which will be 
      4x4 augumented skew-symmetric matrix. The equivalent vector from ``vexa()`` is the twist
      vector (6x1) comprising [v w].


    :seealso: :func:`~trexp`, :func:`~spatialmath.base.transformsNd.vex`, :func:`~spatialmath.base.transformsNd.vexa`
    """
    
    if ishom(T, check=check):
        # SE(3) matrix

        if trn.iseye(T):
            # is identity matrix
            return np.zeros((4,4))
        else:
            [R,t] = trn.tr2rt(T)
            
            if trn.iseye(R):
                # rotation matrix is identity
                skw = np.zeros((3,3))
                v = t
                theta = 1
            else:
                S = trlog(R, check=False)  # recurse
                w = trn.vex(S)
                theta = vec.norm(w)
                skw = trn.skew(w/theta)
                Ginv = np.eye(3) / theta - skw / 2 + (1 / theta - 1 / np.tan(theta / 2) / 2) * skw @ skw
                v = Ginv @ t
            return trn.rt2m(skw, v)*theta
        
    elif isrot(T, check=check):
        # deal with rotation matrix
        R = T
        if trn.iseye(R):
            # matrix is identity
            return np.zeros((3,3))
        elif abs(np.trace(R) + 1) < 100 * _eps:

            # rotation by +/- pi, +/- 3pi etc.
            diagonal = R.diagonal()
            k = diagonal.argmax()
            mx = diagonal[k]
            I = np.eye(3)
            col = R[:,k] + I[:,k]
            w = col / np.sqrt(2 * (1 + mx))
            theta = math.pi
            return trn.skew(w*theta)
        else:
            # general case
            theta = np.arccos((np.trace(R)-1) / 2)
            skw = (R - R.T) / 2 / np.sin(theta)
            return skw * theta
    else:
        raise ValueError("Expect SO(3) or SE(3) matrix")

# ---------------------------------------------------------------------------------------#
def trexp(S, theta=None):
    """
    Exponential of so(3) or se(3) matrix

    :param S: so(3), se(3) matrix or equivalent velctor
    :type T: numpy.ndarray, shape=(3,3), (3,), (4,4), or (6,)
    :param theta: motion
    :type theta: float
    :return: 3x3 or 4x4 matrix exponential in SO(3) or SE(3)
    :rtype: numpy.ndarray, shape=(3,3) or (4,4)
    
    An efficient closed-form solution of the matrix exponential for arguments
    that are so(3) or se(3).
    
    For so(3) the results is an SO(3) rotation matrix:

    - ``trexp(S)`` is the matrix exponential of the so(3) element ``S`` which is a 3x3
       skew-symmetric matrix.
    - ``trexp(S, THETA)`` as above but for an so(3) motion of S*THETA, where ``S`` is
      unit-norm skew-symmetric matrix representing a rotation axis and a rotation magnitude
      given by ``THETA``.
    - ``trexp(W)`` is the matrix exponential of the so(3) element ``W`` expressed as
      a 3-vector (array_like).
    - ``trexp(W, THETA)`` as above but for an so(3) motion of W*THETA where ``W`` is a
      unit-norm vector representing a rotation axis and a rotation magnitude
      given by ``THETA``. ``W`` is expressed as a 3-vector (array_like).


    For se(3) the results is an SE(3) homogeneous transformation matrix:

    - ``trexp(SIGMA)`` is the matrix exponential of the se(3) element ``SIGMA`` which is
      a 4x4 augmented skew-symmetric matrix.
    - ``trexp(SIGMA, THETA)`` as above but for an se(3) motion of SIGMA*THETA, where ``SIGMA``
      must represent a unit-twist, ie. the rotational component is a unit-norm skew-symmetric
      matrix.
    - ``trexp(TW)`` is the matrix exponential of the se(3) element ``TW`` represented as
      a 6-vector which can be considered a screw motion.
    - ``trexp(TW, THETA)`` as above but for an se(3) motion of TW*THETA, where ``TW``
      must represent a unit-twist, ie. the rotational component is a unit-norm skew-symmetric
      matrix.
     
     :seealso: :func:`~trlog, :func:`~spatialmath.base.transforms2d.trexp2`
    """
   
    if argcheck.ismatrix(S, (4,4)) or argcheck.isvector(S, 6):
        # se(3) case
        if argcheck.ismatrix(S, (4,4)):
            # augmentented skew matrix
            tw = trn.vexa(S)
        else:
            # 6 vector
            tw = argcheck.getvector(S)

        if theta is None:
            (tw,theta) = vec.unittwist(tw)
        else:
            assert vec.isunittwist(tw), 'If theta is specified S must be a unit twist'

        t = tw[0:3]
        w = tw[3:6]
        

        R = trn._rodrigues(w, theta)
        
        skw = trn.skew(w)
        V = np.eye(3)*theta + (1.0-math.cos(theta))*skw + (theta-math.sin(theta))*skw @ skw

        return trn.rt2tr(R, V@t)
        
    elif argcheck.ismatrix(S, (3,3)) or argcheck.isvector(S, 3):
        # so(3) case
        if argcheck.ismatrix(S, (3,3)):
            # skew symmetric matrix
            w = trn.vex(S)
        else:
            # 3 vector
            w = argcheck.getvector(S)
            
        if theta is not None:
            assert vec.isunitvec(w), 'If theta is specified S must be a unit twist'

        # do Rodrigues' formula for rotation
        return trn._rodrigues(w, theta)
    else:
        raise ValueError(" First argument must be SO(3), 3-vector, SE(3) or 6-vector")
    

def trprint(T, orient='rpy/zyx', label=None, file=sys.stdout, fmt='{:8.2g}', unit='deg'):
    """
    Compact display of SO(3) or SE(3) matrices
    
    :param T: matrix to format
    :type T: numpy.ndarray, shape=(3,3) or (4,4)
    :param label: text label to put at start of line
    :type label: str
    :param orient: 3-angle convention to use
    :type orient: str
    :param file: file to write formatted string to. [default, stdout]
    :type file: str
    :param fmt: conversion format for each number
    :type fmt: str
    :param unit: angular units: 'rad' [default], or 'deg'
    :type unit: str
    :return: optional formatted string
    :rtype: str
    
    The matrix is formatted and written to ``file`` or if ``file=None`` then the
    string is returned. 

   - ``trprint(R)`` displays the SO(3) rotation matrix in a compact 
      single-line format:
        
        [LABEL:] ORIENTATION UNIT
        
    - ``trprint(T)`` displays the SE(3) homogoneous transform in a compact 
      single-line format:
        
        [LABEL:] [t=X, Y, Z;] ORIENTATION UNIT    
    
    Orientation is expressed in one of several formats:
        
    - 'rpy/zyx' roll-pitch-yaw angles in ZYX axis order [default]
    - 'rpy/yxz' roll-pitch-yaw angles in YXZ axis order
    - 'rpy/zyx' roll-pitch-yaw angles in ZYX axis order
    - 'eul' Euler angles in ZYZ axis order
    - 'angvec' angle and axis
    

    Example:
        
    >>> T = transl(1,2,3) @ rpy2tr(10, 20, 30, 'deg')
    >>> trprint(T, file=None, label='T')
    'T: t =        1,        2,        3; rpy/zyx =       10,       20,       30 deg'
    >>> trprint(T, file=None, label='T', orient='angvec')
    'T: t =        1,        2,        3; angvec = (      56 deg |     0.12,     0.62,     0.78)'
    >>> trprint(T, file=None, label='T', orient='angvec', fmt='{:8.4g}')
    'T: t =        1,        2,        3; angvec = (   56.04 deg |    0.124,   0.6156,   0.7782)'

    Notes:
        
     - If the 'rpy' option is selected, then the particular angle sequence can be
       specified with the options 'xyz' or 'yxz' which are passed through to ``tr2rpy``.
       'zyx' is the default.
     - Default formatting is for readable columns of data
      
    :seealso: :func:`~spatialmath.base.transforms2d.trprint2`, :func:`~tr2eul`, :func:`~tr2rpy`, :func:`~tr2angvec`
    """
    
    s = ''
    
    if label is not None:
        s += '{:s}: '.format(label)
    
    # print the translational part if it exists
    if ishom(T):
        s += 't = {};'.format(_vec2s(fmt, transl(T)))
    
    # print the angular part in various representations

    a = orient.split('/')
    if a[0] == 'rpy':
        if len(a) == 2:
            seq = a[1]
        else:
            seq = None
        angles = tr2rpy(T, order=seq, unit=unit)
        s += ' {} = {} {}'.format(orient, _vec2s(fmt, angles), unit)
        
    elif a[0].startswith('eul'):
        angles = tr2eul(T, unit)
        s += ' eul = {} {}'.format(_vec2s(fmt, angles), unit)
    
    elif a[0] == 'angvec':
        pass
        # as a vector and angle
        (theta,v) = tr2angvec(T, unit)
        if theta == 0:
            s += ' R = nil'
        else:
            s += ' angvec = ({} {} | {})'.format(fmt.format(theta), unit, _vec2s(fmt, v))
    else:
        raise ValueError('bad orientation format')
       
    if file:
        print(s, file=file)
    else:
        return s

    
def _vec2s(fmt, v):
        v = [x if np.abs(x) > 100*_eps else 0.0 for x in v ]
        return ', '.join([fmt.format(x) for x in v])

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    def trplot(T, axes=None, dims=None, color='blue', frame=None, textcolor=None, labels=['X', 'Y', 'Z'], length=1, arrow=True, projection='ortho', rviz=False, wtl=0.2, width=1, d1= 0.05, d2 = 1.15, **kwargs ):
        """
        Plot a 3D coordinate frame
                             
        :param T: an SO(3) or SE(3) pose to be displayed as coordinate frame
        :type: numpy.ndarray, shape=(3,3) or (4,4)
        :param X: the axes to plot into, defaults to current axes
        :type ax: Axes3D reference
        :param dims: dimension of plot volume as [xmin, xmax, ymin, ymax,zmin, zmax]. 
        If dims is [min, max] those limits are applied to the x-, y- and z-axes.
        :type dims: array_like
        :param color: color of the lines defining the frame
        :type color: str
        :param textcolor: color of text labels for the frame, default color of lines above
        :type textcolor: str
        :param frame: label the frame, name is shown below the frame and as subscripts on the frame axis labels
        :type frame: str
        :param labels: labels for the axes, defaults to X, Y and Z
        :type labels: 3-tuple of strings
        :param length: length of coordinate frame axes, default 1
        :type length: float
        :param arrow: show arrow heads, default True
        :type arrow: bool
        :param wtl: width-to-length ratio for arrows, default 0.2
        :type wtl: float
        :param rviz: show Rviz style arrows, default False
        :type rviz: bool
        :param projection: 3D projection: ortho [default] or persp
        :type projection: str
        :param width: width of lines, default 1
        :type width: float
        :param d1: distance of frame axis label text from origin, default 1.15
        :type d2: distance of frame label text from origin, default 0.05
    
        Adds a 3D coordinate frame represented by the SO(3) or SE(3) matrix to the current axes.
        
        - If no current figure, one is created
        - If current figure, but no axes, a 3d Axes is created
        
        Examples:
    
             trplot(T, frame='A')
             trplot(T, frame='A', color='green')
             trplot(T1, 'labels', 'NOA');
    
        """
        
        #TODO
        # animation
        # anaglyph
    
        # check input types
        if isrot(T, check=True):
            T = trn.r2t(T)
        else:
            assert ishom(T, check=True)
    
        if axes is None:
            # create an axes
            fig = plt.gcf()
            if fig.axes == []:
                # no axes in the figure, create a 3D axes
                ax = fig.add_subplot(111, projection='3d', proj_type=projection)
                ax.autoscale(enable=True, axis='both')

                #ax.set_aspect('equal')
                ax.set_xlabel(labels[0])
                ax.set_ylabel(labels[1])
                ax.set_zlabel(labels[2])
            else:
                # reuse an existing axis
                ax = plt.gca()
        else:
            ax = axes
            
        if dims is not None:
            if len(dims) == 2:
                dims = dims * 3
            ax.set_xlim(dims[0:2])
            ax.set_ylim(dims[2:4])
            ax.set_zlim(dims[4:6])
        
        # create unit vectors in homogeneous form
        o =  T @ np.array([0, 0, 0, 1])
        x = T @ np.array([1, 0, 0, 1]) * length
        y = T @ np.array([0, 1, 0, 1]) * length
        z = T @ np.array([0, 0, 1, 1]) * length
    
        # draw the axes
    
        if rviz:
            ax.plot([o[0], x[0]], [o[1], x[1]], [o[2], x[2]], color='red', linewidth=5*width)
            ax.plot([o[0], y[0]], [o[1], y[1]], [o[2], y[2]], color='lime', linewidth=5*width)
            ax.plot([o[0], z[0]], [o[1], z[1]], [o[2], z[2]], color='blue', linewidth=5*width)
        elif arrow:
            ax.quiver(o[0], o[1], o[2], x[0]-o[0], x[1]-o[1], x[2]-o[2], arrow_length_ratio=wtl, linewidth=width, facecolor=color, edgecolor=color)
            ax.quiver(o[0], o[1], o[2], y[0]-o[0], y[1]-o[1], y[2]-o[2], arrow_length_ratio=wtl, linewidth=width, facecolor=color, edgecolor=color)
            ax.quiver(o[0], o[1], o[2], z[0]-o[0], z[1]-o[1], z[2]-o[2], arrow_length_ratio=wtl, linewidth=width, facecolor=color, edgecolor=color)
            # plot an invisible point at the end of each arrow to allow auto-scaling to work
            ax.scatter( xs=[o[0], x[0], y[0], z[0]], ys=[o[1], x[1], y[1], z[1]], zs=[o[2], x[2], y[2], z[2]], s=[20,0,0,0])
        else:
            ax.plot([o[0], x[0]], [o[1], x[1]], [o[2], x[2]], color=color, linewidth=width)
            ax.plot([o[0], y[0]], [o[1], y[1]], [o[2], y[2]], color=color, linewidth=width)
            ax.plot([o[0], z[0]], [o[1], z[1]], [o[2], z[2]], color=color, linewidth=width)
        
        # label the frame
        if frame:
            if textcolor is not None:
                color = textcolor
            
            o1 =  T @ np.array([-d1, -d1, -d1, 1])
            ax.text(o1[0], o1[1], o1[2], '$\{' + frame + '\}$', color=color, verticalalignment='top', horizontalalignment='center')
        
            # add the labels to each axis
            
            x = (x - o) * d2 + o
            y = (y - o) * d2 + o
            z = (z - o) * d2 + o
        
            ax.text(x[0], x[1], x[2], "$%c_{%s}$" % (labels[0],frame), color=color, horizontalalignment='center', verticalalignment='center')
            ax.text(y[0], y[1], y[2], "$%c_{%s}$" % (labels[1], frame),  color=color, horizontalalignment='center', verticalalignment='center')
            ax.text(z[0], z[1], z[2], "$%c_{%s}$" % (labels[2], frame),  color=color, horizontalalignment='center', verticalalignment='center')

except:  # pragma: no cover
    def trplot(*args, **kwargs):
        print('** trplot: no plot produced -- matplotlib not installed')
    
if __name__ == '__main__':  # pragma: no cover
    import pathlib
    import os.path
    

    
    # trplot( transl(1,2,3), frame='A', rviz=True, width=1, dims=[0, 10, 0, 10, 0, 10])
    # trplot( transl(3,1, 2), color='red', width=3, frame='B')
    # trplot( transl(4, 3, 1)@trotx(math.pi/3), color='green', frame='c', dims=[0,4,0,4,0,4])
    
    exec(open(os.path.join(pathlib.Path(__file__).parent.absolute(), "test_transforms.py")).read() )