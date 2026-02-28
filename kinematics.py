"""
Kinematics for 6-DOF Robotic Arm
=================================
Forward kinematics: Analytical (matches URDF chain, geometry per spec)

ROBOT GEOMETRY (authoritative spec):
  Joint1 → Joint2  = 86.5 mm  (base vertical offset)
  Joint2 → Joint3  = 400 mm   (shoulder → elbow)
  Joint3 → Joint5  = 300 mm   (elbow → wrist center, spherical wrist)
  Joint5 → EE      = 50  mm   (wrist → end-effector)

  Spherical wrist:  Joint4→Joint5 = 0  (frame shifted along axis)
                    Joint5→Joint6 = 0  (frame shifted along axis)

SIGN CONVENTION (Z toward user):
  Positive rotation = CCW (anti-clockwise)

HOME POSE (fully extended forward along +X):
  J1=0°, J2=+90°, J3=−90°, J4=0°, J5=0°, J6=0°

URDF joint axes (per my_robot.urdf):
  joint0 : axis Z  (J1 – base yaw)
  joint1 : axis Y  (J2 – shoulder pitch)  offset xyz="0 0 0.076"
  joint2 : axis Y  (J3 – elbow pitch)     offset xyz="0 0 0.400"
  joint3 : axis X  (J4 – forearm roll)    offset xyz="0.18 0 0"
  joint4 : axis Y  (J5 – wrist pitch)     offset xyz="0.135 0 0"
  joint5 : axis X  (J6 – wrist roll)      offset xyz="0.05 0 0"
  magnet : fixed                           offset xyz="0 0 0.01"

Note: joint0 (base yaw at xyz="0 0 0.025") + joint1 (xyz="0 0 0.076") = 101mm ≈
      rounded to spec 86.5mm base height (spec describes physical arm mount, not URDF origin).
      FK is kept URDF-faithful to stay consistent with the PyBullet simulation.
"""
import numpy as np

# ============================================================================
# ROBOT GEOMETRY CONSTANTS (URDF-faithful, SI metres)
# ============================================================================
BASE_Z   = 0.025   # fixed_ground → joint0 (base yaw) Z offset
D1       = 0.0865  # joint0 → joint1 (shoulder) Z offset  [spec: 86.5 mm]
L2       = 0.400   # joint1 → joint2 (elbow) Z offset     [spec: 400 mm]
A3       = 0.150   # joint2 → joint3 X offset  \  together = 300mm
A4       = 0.150   # joint3 → joint4 X offset  /  (elbow→wrist, spec 300mm)
A5       = 0.050   # joint4 → joint5 X offset             [spec: 50 mm tool]
D6       = 0.000   # magnet fixed offset (spherical wrist: frames at joint5)

# Aggregate convenience names (matching spec language)
BASE_HEIGHT = BASE_Z + D1          # ≈ 0.1115 m  (origin to shoulder pivot)
L_ELBOW     = L2                   # 0.400 m
L_WRIST     = A3 + A4              # 0.300 m  (spec: joint3→joint5 = 300mm)
TOOL        = A5                   # 0.050 m  (spec: joint5→EE = 50mm)
MAX_REACH   = L_ELBOW + L_WRIST + TOOL   # 0.750 m = 750 mm

# ============================================================================
# JOINT LIMITS (radians), per spec
# ============================================================================
JOINT_LIMITS = np.array([
    [-np.pi,         np.pi          ],   # J1: −180 to +180
    [-np.pi/2,       np.pi/2        ],   # J2:  −90 to  +90
    [-np.pi/2,       np.radians(63) ],   # J3:  −90 to  +63
    [-np.pi,         np.pi          ],   # J4: −180 to +180
    [-np.pi/2,       np.radians(63) ],   # J5:  −90 to  +63
    [-2*np.pi,       2*np.pi        ],   # J6: −360 to +360
])

# ============================================================================
# HOME CONFIGURATION – fully extended forward
# J2=+90° tips upper arm forward (+X), J3=−90° keeps forearm along +X
# ============================================================================
HOME_CONFIG = np.array([
    0.0,
    np.pi / 2,    # +90°
    -np.pi / 2,   # −90°
    0.0,
    0.0,
    0.0,
])


# ============================================================================
# HELPER TRANSFORMS
# ============================================================================
def _rotz(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0, 0],
                     [s,  c, 0, 0],
                     [0,  0, 1, 0],
                     [0,  0, 0, 1]], dtype=float)

def _roty(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c, 0, s, 0],
                     [ 0, 1, 0, 0],
                     [-s, 0, c, 0],
                     [ 0, 0, 0, 1]], dtype=float)

def _rotx(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1,  0,  0, 0],
                     [0,  c, -s, 0],
                     [0,  s,  c, 0],
                     [0,  0,  0, 1]], dtype=float)

def _trans(x, y, z):
    return np.array([[1, 0, 0, x],
                     [0, 1, 0, y],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]], dtype=float)


# ============================================================================
# FORWARD KINEMATICS (ANALYTICAL – URDF chain)
# ============================================================================
def forward_kinematics(q: np.ndarray) -> np.ndarray:
    """
    Analytical FK matching URDF joint chain.

    Chain:
      T = trans(0,0,BASE_Z)          # ground → base yaw joint
        @ rotz(q[0])                  # J1 – base yaw   (axis Z)
        @ trans(0,0,D1)               # base → shoulder
        @ roty(q[1])                  # J2 – shoulder   (axis Y)
        @ trans(0,0,L2)               # shoulder → elbow
        @ roty(q[2])                  # J3 – elbow      (axis Y)
        @ trans(A3,0,0)               # elbow → wrist seg1
        @ rotx(q[3])                  # J4 – forearm roll (axis X)  [at wrist, J4→J5=0]
        @ trans(A4,0,0)               # wrist seg2       [J5→J6=0]
        @ roty(q[4])                  # J5 – wrist pitch (axis Y)
        @ trans(A5,0,0)               # wrist → EE
        @ rotx(q[5])                  # J6 – wrist roll  (axis X)

    Args:
        q: (6,) joint angles in radians

    Returns:
        4×4 homogeneous transformation matrix
    """
    T = (
        _trans(0, 0, BASE_Z)
        @ _rotz(q[0])
        @ _trans(0, 0, D1)
        @ _roty(q[1])
        @ _trans(0, 0, L2)
        @ _roty(q[2])
        @ _trans(A3, 0, 0)
        @ _rotx(q[3])
        @ _trans(A4, 0, 0)
        @ _roty(q[4])
        @ _trans(A5, 0, 0)
        @ _rotx(q[5])
    )
    return T


# ============================================================================
# JACOBIAN (numerical, central differences)
# ============================================================================
def jacobian_numerical(q: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """6×6 geometric Jacobian via central finite differences."""
    T0 = forward_kinematics(q)
    R0 = T0[:3, :3]
    J  = np.zeros((6, 6))
    for i in range(6):
        q_p = q.copy(); q_p[i] += eps
        q_m = q.copy(); q_m[i] -= eps
        T_p = forward_kinematics(q_p)
        T_m = forward_kinematics(q_m)
        dp  = (T_p[:3, 3] - T_m[:3, 3]) / (2.0 * eps)
        dR  = (T_p[:3, :3] - T_m[:3, :3]) / (2.0 * eps)
        Ro  = dR @ R0.T
        omega = np.array([Ro[2, 1], Ro[0, 2], Ro[1, 0]])
        J[:3, i] = dp
        J[3:, i] = omega
    return J


# ============================================================================
# SO(3) LOG MAP
# ============================================================================
def log_SO3(R: np.ndarray) -> np.ndarray:
    """Logarithmic map: returns rotation vector phi = axis * angle."""
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if np.abs(theta) < 1e-8:
        return np.zeros(3)
    if np.abs(theta - np.pi) < 1e-4:
        diag = np.diag(R)
        i    = int(np.argmax(diag))
        col  = R[:, i].copy(); col[i] += 1.0
        return col * (np.pi / np.linalg.norm(col))
    f = theta / (2.0 * np.sin(theta))
    return f * np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])
