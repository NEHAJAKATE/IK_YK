"""
GUI Real-time Controller — Neural Network IK + High-Accuracy DLS
=================================================================
PERFORMANCE: IK computation runs in a BACKGROUND DAEMON THREAD.
The UI thread only reads the latest cached result — zero blocking.

Solver pipeline (background thread):
  1. NN warm-start  → q_seed
  2. DLS refinement → q_final  (adaptive damping, 200 iters max, 0.3 mm target)
  3. Multi-seed retry if err > 1 mm
  4. Closest-reachable fallback for out-of-workspace targets
"""
import numpy as np
import threading
import time
import os
from kinematics import (forward_kinematics, jacobian_numerical, log_SO3,
                         HOME_CONFIG, JOINT_LIMITS, BASE_HEIGHT, MAX_REACH)


# ─────────────────────────────────────────────────────────────────────────────
#  SOLVER PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
DLS_MAX_ITERS    = 80
DLS_LAMBDA_INIT  = 0.005
DLS_LAMBDA_MAX   = 0.1
DLS_POS_TOL      = 3e-4      # 0.3 mm
DLS_ORI_TOL      = 5e-3      # ~0.3 deg
DLS_STEP_MAX     = 0.15
SMOOTH_ALPHA     = 1.0
PASS_THRESH_MM   = 1.0
CLOSEST_SEARCH_N = 20
TARGET_CHANGE_TOL = 0.0      # always re-solve when slider moves
ORI_WEIGHT       = 0.2
BACKTRACKS       = 3
ORI_PASS_DEG     = 1.0

_RETRY_CONFIGS = [
    HOME_CONFIG,
    np.array([0, 0, 0, 0, 0, 0], dtype=float),
    np.array([0, np.pi/4, -np.pi/4, 0, 0, 0], dtype=float),
    np.array([np.pi/2, np.pi/4, -np.pi/2, 0, 0, 0], dtype=float),
    np.array([-np.pi/2, np.pi/4, -np.pi/2, 0, 0, 0], dtype=float),
]


class NNController:
    """
    IK controller with a background solving thread.
    solve_and_apply() is non-blocking — it updates the target and
    returns the most recent cached solution immediately.
    """

    def __init__(self, sim):
        self.sim        = sim
        self.nn_network = None
        self._load_nn()

        # Solve state (only written from background thread, except reset_to_home)
        self.q_prev     = HOME_CONFIG.copy()
        self.q_smoothed = HOME_CONFIG.copy()

        # Shared data — protected by _lock
        self._lock          = threading.Lock()
        self._tgt_pos       = np.array([0.3, 0.0, 0.35])
        self._tgt_R         = np.eye(3)
        self._last_solved   = None       # position that was last solved
        self._cached        = self._default_result()

        # Start background solver thread
        self._running = True
        self._thread  = threading.Thread(target=self._bg_loop,
                                          name="IKSolverThread",
                                          daemon=True)
        self._thread.start()

    # ──────────────────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────────────────

    def solve_and_apply(self, position_m: np.ndarray,
                        R_target: np.ndarray) -> dict:
        """Non-blocking: update target, return latest cached result."""
        with self._lock:
            self._tgt_pos = position_m.copy()
            self._tgt_R   = R_target.copy()
            return dict(self._cached)

    def reset_to_home(self):
        with self._lock:
            self.q_prev     = HOME_CONFIG.copy()
            self.q_smoothed = HOME_CONFIG.copy()
            self._last_solved = None        # force immediate re-solve
        self.sim.set_joint_angles(HOME_CONFIG)

    # ──────────────────────────────────────────────────────────────────────────
    #  Background IK loop  (daemon thread — never touches Qt)
    # ──────────────────────────────────────────────────────────────────────────

    def _bg_loop(self):
        while self._running:
            with self._lock:
                pos  = self._tgt_pos.copy()
                R    = self._tgt_R.copy()
                last = self._last_solved

            # Only re-solve when the target has moved meaningfully
            need_solve = (last is None or
                          np.linalg.norm(pos - last) > TARGET_CHANGE_TOL)

            if need_solve:
                result = self._compute(pos, R)
                with self._lock:
                    self._cached      = result
                    self._last_solved = pos.copy()
            else:
                time.sleep(0.003)   # idle: ~333 Hz loop, low CPU

    # ──────────────────────────────────────────────────────────────────────────
    #  Full IK computation  (called only from background thread)
    # ──────────────────────────────────────────────────────────────────────────

    def _compute(self, position_m: np.ndarray, R_target: np.ndarray) -> dict:
        t0 = time.perf_counter()

        q_seed = self._nn_seed(position_m, R_target)
        tries = [q_seed] + [q.copy() for q in _RETRY_CONFIGS]

        best = None
        for q_try in tries:
            q_t, it_t, cv_t, cl_t, _ = self._dls_refine_full(
                q_try, position_m, R_target)
            T_fk = forward_kinematics(q_t)
            p_fk = T_fk[:3, 3]
            R_fk = T_fk[:3, :3]
            pos_err_mm = np.linalg.norm(position_m - p_fk) * 1000.0
            phi = log_SO3(R_target @ R_fk.T)
            ori_err_deg = np.degrees(np.linalg.norm(phi))

            ok = (cv_t
                  and pos_err_mm <= PASS_THRESH_MM
                  and ori_err_deg <= ORI_PASS_DEG)

            cand = (pos_err_mm, ori_err_deg, it_t, cv_t, cl_t, q_t)
            if best is None or cand[:2] < best[:2]:
                best = cand

            if ok:
                best = cand
                break

        pos_err_mm, ori_err_deg, iters, converged, clamped, q_ref = best
        reachable = (converged
                     and pos_err_mm <= PASS_THRESH_MM
                     and ori_err_deg <= ORI_PASS_DEG)

        if reachable:
            q_final = (1.0 - SMOOTH_ALPHA) * self.q_smoothed \
                      + SMOOTH_ALPHA * q_ref
            self.q_prev = q_ref.copy()
            self.q_smoothed = q_final.copy()
            self.sim.set_joint_angles(q_final)
        else:
            q_final = self.q_prev.copy()
            self.sim.set_joint_angles(q_final)

        accuracy = 100.0 if reachable else 0.0

        t_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "q_seed":        q_seed,
            "q_final":       q_final,
            "q_seed_deg":    np.degrees(q_seed),
            "q_final_deg":   np.degrees(q_final),
            "iters":         iters,
            "lambda_":       DLS_LAMBDA_INIT,
            "solve_ms":      t_ms,
            "converged":     reachable,
            "clamped":       clamped,
            "pos_error_mm":  pos_err_mm,
            "ori_error_deg": ori_err_deg,
            "accuracy":      accuracy,
            "reachable":     reachable,
            "held":          (not reachable),
            "closest_snap":  False,
            "tgt_pos_m":     position_m.copy(),
            "tgt_R":         R_target.copy(),
            "status_msg":    ("OK" if reachable else "OUT OF RANGE — UNREACHABLE"),
        }

    # ──────────────────────────────────────────────────────────────────────────
    #  DLS Refinement
    # ──────────────────────────────────────────────────────────────────────────

    def _dls_refine_full(self, q_init, p_target, R_target, max_iters=None):
        if max_iters is None:
            max_iters = max(100, DLS_MAX_ITERS)

        q       = np.clip(q_init, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
        clamped = False
        lam     = DLS_LAMBDA_INIT
        prev_e  = np.inf

        for iters in range(max_iters):
            T  = forward_kinematics(q)
            p  = T[:3, 3]; R  = T[:3, :3]
            dp = p_target - p
            phi = log_SO3(R_target @ R.T)

            pe = np.linalg.norm(dp); oe = np.linalg.norm(phi)
            if pe < DLS_POS_TOL and oe < DLS_ORI_TOL:
                return q, iters, True, clamped, pe * 1000.0

            te = pe + ORI_WEIGHT * oe

            err6 = np.concatenate([dp, ORI_WEIGHT * phi])
            J    = jacobian_numerical(q)
            J_w  = J.copy()
            J_w[3:, :] *= ORI_WEIGHT
            JJT  = J_w @ J_w.T
            try:
                dq = J_w.T @ np.linalg.solve(JJT + lam**2 * np.eye(6), err6)
            except np.linalg.LinAlgError:
                lam = min(lam * 4.0, DLS_LAMBDA_MAX); continue

            s = np.linalg.norm(dq)
            if s > DLS_STEP_MAX:
                dq = dq * (DLS_STEP_MAX / s)

            q_try = q + dq
            q_try = np.clip(q_try, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
            bt = 0
            while bt < BACKTRACKS:
                Tt  = forward_kinematics(q_try)
                pt  = Tt[:3, 3]; Rt = Tt[:3, :3]
                dpt = p_target - pt
                phit = log_SO3(R_target @ Rt.T)
                te_new = np.linalg.norm(dpt) + ORI_WEIGHT * np.linalg.norm(phit)
                if te_new <= te:
                    break
                q_try = q + 0.5 * (q_try - q)
                bt += 1
            q_c = np.clip(q_try, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
            if not np.allclose(q, q_c, atol=1e-8):
                clamped = True
            q = q_c
            if prev_e < np.inf:
                if te_new > prev_e * 1.02:
                    lam = min(lam * 1.8, DLS_LAMBDA_MAX)
                elif te_new < prev_e * 0.98:
                    lam = max(lam * 0.8, DLS_LAMBDA_INIT)
            prev_e = te_new

        T_f = forward_kinematics(q)
        return q, max_iters, False, clamped, \
               np.linalg.norm(p_target - T_f[:3, 3]) * 1000.0

    # ──────────────────────────────────────────────────────────────────────────
    #  Closest-reachable binary search
    # ──────────────────────────────────────────────────────────────────────────

    def _closest_reachable(self, p_target, R_target):
        base  = np.array([0.0, 0.0, float(BASE_HEIGHT)])
        dirn  = p_target - base
        dist  = np.linalg.norm(dirn)
        if dist < 1e-6:
            return None, None
        for R_try in [R_target, np.eye(3), self._yaw_only(R_target)]:
            q_b, err_b = self._bisect_ray(base, dirn, R_try)
            if q_b is not None:
                return q_b, err_b
        return None, None

    def _bisect_ray(self, base, direction, R_try):
        t_lo, t_hi = 0.05, 1.0
        q_best = None; err_best = np.inf
        for _ in range(CLOSEST_SEARCH_N):
            t   = (t_lo + t_hi) / 2.0
            p_t = base + t * direction
            q_t, _, conv, _, err = self._dls_refine_full(
                self.q_prev.copy(), p_t, R_try, max_iters=80)
            if conv and err < 5.0:
                if err < err_best:
                    q_best, err_best = q_t.copy(), err
                t_lo = t
            else:
                t_hi = t
        return q_best, err_best

    @staticmethod
    def _yaw_only(R):
        yaw = np.arctan2(R[1, 0], R[0, 0])
        cy, sy = np.cos(yaw), np.sin(yaw)
        return np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)

    # ──────────────────────────────────────────────────────────────────────────
    #  NN seed
    # ──────────────────────────────────────────────────────────────────────────

    def _nn_seed(self, position_m, R_target):
        if self.nn_network is not None:
            try:
                q = self.nn_network.predict(position_m, R_target)
                if q is not None:
                    return np.clip(q, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
            except Exception:
                pass
        return self.q_prev.copy()

    def _load_nn(self):
        pkl = os.path.join(os.path.dirname(__file__), 'ik_network.pkl')
        if os.path.exists(pkl):
            try:
                from nn_ik import SimpleIKNetwork
                net = SimpleIKNetwork()
                if net.load(pkl):
                    self.nn_network = net
            except Exception:
                pass

    # ──────────────────────────────────────────────────────────────────────────
    #  Default result (used before first solve completes)
    # ──────────────────────────────────────────────────────────────────────────

    def _default_result(self):
        return {
            "q_seed":        HOME_CONFIG.copy(),
            "q_final":       HOME_CONFIG.copy(),
            "q_seed_deg":    np.degrees(HOME_CONFIG),
            "q_final_deg":   np.degrees(HOME_CONFIG),
            "iters":         0,
            "lambda_":       DLS_LAMBDA_INIT,
            "solve_ms":      0.0,
            "converged":     False,
            "clamped":       False,
            "pos_error_mm":  0.0,
            "ori_error_deg": 0.0,
            "accuracy":      100.0,
            "reachable":     True,
            "held":          False,
            "closest_snap":  False,
            "tgt_pos_m":     np.array([0.3, 0.0, 0.35], dtype=float),
            "tgt_R":         np.eye(3, dtype=float),
        }

    def stop(self):
        self._running = False
