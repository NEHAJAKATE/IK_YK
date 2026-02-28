"""
IK accuracy test – no PyBullet needed.
Tests 50 random reachable configs: FK → target → DLS solve → error.
"""
import sys, types, io
sys.path.insert(0, '.')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.modules.setdefault('pybullet',      types.ModuleType('pybullet'))
sys.modules.setdefault('pybullet_data', types.ModuleType('pybullet_data'))

import numpy as np, math

class FakeSim:
    def set_joint_angles(self, q): pass

from kinematics import (forward_kinematics, JOINT_LIMITS, HOME_CONFIG, log_SO3)
from gui_control_nn import NNController, DLS_MAX_ITERS, DLS_POS_TOL

ctrl = NNController(FakeSim())

np.random.seed(2024)
errors = []
passes = 0
N = 50

print(f"Testing {N} random reachable targets...")
print(f"DLS: max_iters={DLS_MAX_ITERS}, tol={DLS_POS_TOL*1000:.1f}mm\n")

for i in range(N):
    # Generate reachable target via FK from random q (within limits, away from singularities)
    q_true = np.random.uniform(JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
    # Keep J2/J3 in upper half to avoid singularities
    q_true[1] = np.random.uniform(0.1, np.pi/2)
    q_true[2] = np.random.uniform(-np.pi/2, 0.1)

    T_true = forward_kinematics(q_true)
    p_tgt  = T_true[:3, 3]
    R_tgt  = T_true[:3, :3]

    # Solve via controller
    r = ctrl.solve_and_apply(p_tgt, R_tgt)
    e = r['pos_error_mm']
    errors.append(e)
    if e < 1.0:
        passes += 1

    if (i+1) % 10 == 0:
        avg = np.mean(errors[-10:])
        mx  = max(errors[-10:])
        print(f"  [{i+1:3d}/{N}]  last-10: avg={avg:.3f}mm  max={mx:.3f}mm  "
              f"pass_rate={passes}/{i+1}={100*passes/(i+1):.1f}%")

print()
errors = np.array(errors)
print("=" * 55)
print(f"  RESULTS over {N} targets")
print(f"  Mean error : {np.mean(errors):.4f} mm")
print(f"  Median err : {np.median(errors):.4f} mm")
print(f"  Max error  : {np.max(errors):.4f} mm")
print(f"  95th pctile: {np.percentile(errors, 95):.4f} mm")
print(f"  Pass (<1mm): {passes}/{N} = {100*passes/N:.1f}%")
print("=" * 55)

ok = passes / N >= 0.95
sys.exit(0 if ok else 1)
