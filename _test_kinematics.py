"""
Headless import check – no PyBullet needed.
"""
import sys, types
sys.path.insert(0, '.')
sys.modules.setdefault('pybullet',      types.ModuleType('pybullet'))
sys.modules.setdefault('pybullet_data', types.ModuleType('pybullet_data'))
import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import math, numpy as np

# =====================================
# 1. kinematics
# =====================================
from kinematics import (forward_kinematics, HOME_CONFIG, JOINT_LIMITS,
                         log_SO3, jacobian_numerical,
                         BASE_Z, D1, L2, A3, A4, A5, D6,
                         BASE_HEIGHT, L_ELBOW, L_WRIST, TOOL, MAX_REACH)

ok = True
def chk(c, msg, ex=""):
    global ok
    s = "PASS" if c else "FAIL"
    if not c: ok = False
    print(f"  [{s}]  {msg}  {ex}")

print("=== Geometry constants ===")
chk(abs(D1 - 0.0865) < 1e-9, "D1 = 86.5 mm", f"got {D1*1000:.2f}")
chk(abs(L2 - 0.400)  < 1e-9, "L2 = 400 mm",  f"got {L2*1000:.1f}")
chk(abs(A3 - 0.150)  < 1e-9, "A3 = 150 mm",  f"got {A3*1000:.1f}")
chk(abs(A4 - 0.150)  < 1e-9, "A4 = 150 mm",  f"got {A4*1000:.1f}")
chk(abs(A5 - 0.050)  < 1e-9, "A5 = 50 mm",   f"got {A5*1000:.1f}")
chk(abs(L_WRIST - 0.300) < 1e-9, "L_WRIST = A3+A4 = 300 mm", f"got {L_WRIST*1000:.1f}")
chk(abs(MAX_REACH - 0.750) < 1e-9, "MAX_REACH = 750 mm", f"got {MAX_REACH*1000:.1f}")

print("\n=== HOME_CONFIG ===")
expected_home = [0, math.pi/2, -math.pi/2, 0, 0, 0]
for i in range(6):
    chk(abs(HOME_CONFIG[i] - expected_home[i]) < 1e-9,
        f"J{i+1} = {math.degrees(expected_home[i]):.0f} deg",
        f"got {math.degrees(HOME_CONFIG[i]):.2f}")

print("\n=== FK at home ===")
T = forward_kinematics(HOME_CONFIG)
p = T[:3, 3]
print(f"  EE: X={p[0]*1000:.2f}  Y={p[1]*1000:.4f}  Z={p[2]*1000:.2f}  mm")
chk(abs(p[1]) < 1e-3, "Y ~ 0", f"got {p[1]*1000:.4f}")

print("\n=== Joint limits ===")
lims = [(-180,180),(-90,90),(-90,63),(-180,180),(-90,63),(-360,360)]
for i,(lo,hi) in enumerate(lims):
    el=math.radians(lo); eh=math.radians(hi)
    chk(abs(JOINT_LIMITS[i,0]-el)<1e-9 and abs(JOINT_LIMITS[i,1]-eh)<1e-9,
        f"J{i+1} [{lo},{hi}] deg")

print("\n=== log_SO3 ===")
chk(np.linalg.norm(log_SO3(np.eye(3))) < 1e-10, "log_SO3(I) = 0 vector")

print("\n=== Jacobian ===")
J = jacobian_numerical(HOME_CONFIG)
chk(J.shape==(6,6), "Shape (6,6)")
chk(not np.any(np.isnan(J)), "No NaN")

# =====================================
# 2. NNController import
# =====================================
print("\n=== NNController import ===")
try:
    # Stub sim
    class FakeSim:
        def set_joint_angles(self, q): pass
    from gui_control_nn import NNController
    ctrl = NNController(FakeSim())
    chk(True, "NNController created")
    chk(hasattr(ctrl, 'solve_and_apply'), "solve_and_apply exists")
    chk(hasattr(ctrl, '_closest_reachable'), "_closest_reachable exists")
except Exception as e:
    chk(False, f"NNController import: {e}")

print()
print("=============================================")
print(f"  {'ALL PASS' if ok else 'SOME FAILED'}")
print("=============================================")
sys.exit(0 if ok else 1)
