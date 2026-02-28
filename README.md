# 6-DOF Robotic Arm Digital Twin

Industry-grade real-time simulation, neural-network IK, and control demo.

## Quick Start

```bash
cd final_KPIT_NN
python main_gui.py
```

---

## Robot Geometry

| Parameter | Value |
|---|---|
| Base height (Z offset) | **100 mm** |
| Shoulder → Elbow (Link 2) | **400 mm** |
| Elbow → Wrist center (Link 3) | **300 mm** |
| Wrist center → End-effector | **50 mm** |
| **Total max horizontal reach** | **750 mm** |

### Spherical Wrist Assumption

Joints 4, 5, 6 form a **spherical wrist** — their axes intersect at a single point (the wrist center).  
In the DH/analytical model this means:

- **J4 → J5 offset = 0** (pure rotation about wrist X-axis)
- **J5 → J6 offset = 0** (pure rotation about wrist Y-axis)

The wrist center is placed at the end of Link 3 (300 mm from the elbow joint).  
The 50 mm tool offset is applied along Z *after* all wrist rotations.

---

## Sign Convention

> **Z axis points toward the user (out of screen in the standard view).**

| Rule |
|---|
| Positive rotation = **Counter-Clockwise (CCW)** when viewed from the positive axis direction |

### Home Pose (Fully Extended Forward)

| Joint | Angle |
|---|---|
| J1 | 0° |
| J2 | **+90°** |
| J3 | **−90°** |
| J4 | 0° |
| J5 | 0° |
| J6 | 0° |

At this pose the end-effector is at approximately **(750 mm, 0, 100 mm)** – fully extended along the robot's X-forward axis at base height.

---

## Joint Limits

| Joint | Min | Max |
|---|---|---|
| J1 (base yaw) | −180° | +180° |
| J2 (shoulder pitch) | −90° | +90° |
| J3 (elbow pitch) | −90° | **+63°** |
| J4 (forearm roll) | −180° | +180° |
| J5 (wrist pitch) | −90° | **+63°** |
| J6 (wrist roll) | −360° | +360° |

---

## IK Algorithm

The solver is a **hybrid NN + DLS pipeline**:

1. **NN Warm-Start** – A pre-trained `SimpleIKNetwork` predicts an initial joint configuration `q_seed` from the target pose (position + rotation matrix).
2. **DLS Refinement** – Damped Least Squares Jacobian iteration refines `q_seed` → `q_final`.  
   - λ (damping) = 0.01  
   - Max iterations = 50  
   - Convergence threshold: Δposition < 1 mm, Δrot < 0.01 rad  
3. **Exponential Smoothing** – `q_smooth = 0.7·q_prev + 0.3·q_new` (disabled when position error > 1 mm).
4. **Fail-Safe Hold** – If the refined pose exceeds the 1 mm position-error threshold OR violates joint limits, the robot **holds the last valid configuration** and the UI shows `UNREACHABLE – Holding Last Valid Pose`.

---

## Error Computation (Industry Standard)

After every IK solve, FK is run on `q_final` to measure actual accuracy:

```
T_fk  = forward_kinematics(q_final)

# Position error
err_pos_mm = ||p_target − T_fk[:3,3]|| × 1000

# Orientation error via SO(3) logarithmic map
R_err       = R_target × T_fk[:3,:3]ᵀ
phi         = log_SO3(R_err)          # rotation vector
err_ori_deg = ||phi|| × 180/π

# Accuracy
accuracy    = max(0, 100 × (1 − err_pos_mm / 1.0))

# PASS criterion
PASS if err_pos_mm ≤ 1.0 mm
```

---

## UI Layout

| Region | Content |
|---|---|
| **Left panel** | X/Y/Z sliders (mm), Roll/Pitch/Yaw sliders (°), Reset Home, Random Pose, Demo Mode toggle |
| **Right panel** | Joint angles (refined + NN seed), position error, orientation error, accuracy %, solver diagnostics |
| **Bottom strip** | REACHABLE / UNREACHABLE badge, LIMIT CLAMP indicator, last-valid timestamp |

### Safe Demo Working Region (slider bounds)

| Axis | Min | Max |
|---|---|---|
| X | 100 mm | 600 mm |
| Y | −400 mm | +400 mm |
| Z | 150 mm | 700 mm |

---

## File Structure

| File | Role |
|---|---|
| `main_gui.py` | Entry point – wires simulation, controller, and GUI |
| `ui.py` | PyQt6 dark-theme GUI, timer loop, telemetry display |
| `gui_control_nn.py` | `NNController` class – NN seed + DLS refinement + fail-safe |
| `kinematics.py` | Analytical FK, Jacobian (numerical), `log_SO3`, joint limits |
| `nn_ik.py` | `SimpleIKNetwork` – training, saving, loading, prediction |
| `simulation_threaded.py` | Threaded PyBullet wrapper |
| `generate_training_data.py` | Generate random FK poses for NN training |

### Training the NN (optional)

```bash
python generate_training_data.py   # creates ik_training_data.pkl
python nn_ik.py                    # trains and saves ik_network.pkl
python main_gui.py                 # NN is auto-loaded at startup
```

If `ik_network.pkl` is absent, the controller falls back to DLS-only mode.
