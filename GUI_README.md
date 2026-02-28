# 6-DOF Robot Control - Professional GUI Application

## Overview

Professional PyQt6-based GUI for real-time control of a 6-DOF robotic arm with PyBullet simulation.

## Architecture

### Non-Blocking Design

**Problem Solved:** The CLI version had blocking input and "Unknown command: >" parsing errors.

**Solution:**
- **GUI Thread:** Handles UI updates at 30 Hz using QTimer
- **Simulation Thread:** Runs PyBullet physics at 240 Hz in background
- **Thread-Safe Communication:** All state access protected by locks
- **No Blocking:** User can interact with UI while robot moves

### Components

```
main_gui.py              # Application entry point
ui.py                    # PyQt6 GUI (left panel + controls)
simulation_threaded.py   # Threaded PyBullet wrapper
kinematics.py            # Forward kinematics
```

## Installation

```bash
pip install numpy pybullet PyQt6
```

## Usage

```bash
python main_gui.py
```

### GUI Layout

**LEFT PANEL (35%):**
- Status indicator (READY/MOVING/ERROR)
- Target input fields (cm and degrees)
- Control buttons (Move, Home, Stop, Random Test)
- Current state display (joints, position, orientation)
- Error metrics (position/orientation errors with color coding)
- Scrollable logs (timestamped, color-coded)

**RIGHT SIDE (65%):**
- PyBullet 3D simulation window (separate window)
- Real-time robot visualization

### Controls

**Target Input:**
- Position: X, Y, Z in **centimeters**
- Orientation: Roll, Pitch, Yaw in **degrees**

**Buttons:**
- **Move:** Execute smooth trajectory to target
- **Home:** Return to home position [0,0,0,0,0,0]
- **Stop:** Immediately halt current motion
- **Random Test:** Generate random reachable target

### Error Display

- **Position Error:** < 1.0 cm = GREEN, > 1.0 cm = RED
- **Orientation Error:** < 5.0° = GREEN, > 5.0° = RED
- **Status Badge:** PASS/FAIL based on tolerances

### Logs

- **Green:** Successful operations
- **Red:** Errors and warnings
- **Timestamps:** All entries timestamped

## How It Works

### 1. Input Conversion

```python
# User enters: 40 cm, 0 cm, 30 cm
# GUI converts: 0.4 m, 0.0 m, 0.3 m

# User enters: 0°, 45°, 90°
# GUI converts: 0 rad, 0.785 rad, 1.571 rad
```

### 2. IK Computation

```python
# Convert Euler angles → Quaternion
target_quat = quaternion_from_euler(roll, pitch, yaw)

# PyBullet IK
solution = p.calculateInverseKinematics(
    robot_id, ee_link, target_pos, target_quat,
    maxNumIterations=100, residualThreshold=1e-5
)
```

### 3. Smooth Trajectory

```python
# Generate interpolated path (60 steps)
trajectory = [current_q + (target_q - current_q) * t 
              for t in linspace(0, 1, 60)]

# Execute in simulation thread at 240 Hz
# UI remains responsive during motion
```

### 4. Error Computation

**Position Error:**
```python
error_m = ||pos_actual - pos_target||
error_cm = error_m * 100
```

**Orientation Error:**
```python
# Quaternion angular distance
dot = |q_actual · q_target|
angle_rad = 2 * arccos(dot)
angle_deg = degrees(angle_rad)
```

## Threading Model

```
┌─────────────────┐         ┌──────────────────┐
│   GUI Thread    │         │ Simulation Thread│
│   (30 Hz)       │         │   (240 Hz)       │
├─────────────────┤         ├──────────────────┤
│ • QTimer        │         │ • Physics step   │
│ • Update UI     │◄───────►│ • Trajectory exec│
│ • Handle input  │  Lock   │ • State update   │
│ • Display state │         │ • IK computation │
└─────────────────┘         └──────────────────┘
```

**Key Points:**
- GUI never blocks waiting for simulation
- Simulation runs continuously in background
- State access synchronized with locks
- Stop button works immediately (sets flag checked in sim thread)

## Advantages Over CLI

| CLI Version | GUI Version |
|-------------|-------------|
| Blocking input | Non-blocking, responsive |
| "Unknown command: >" errors | Validated input fields |
| Manual typing | Click buttons, sliders |
| No visual feedback | Real-time error display |
| Text-only logs | Color-coded, timestamped logs |
| No status indicator | Live status (READY/MOVING/ERROR) |

## Troubleshooting

**PyBullet window not visible:**
- Check if window opened behind other windows
- Window should appear automatically when GUI starts

**Robot not moving:**
- Check logs for error messages
- Verify target is within workspace (X: 30-60cm, Y: -20-20cm, Z: 20-50cm)
- Check joint limits not exceeded

**High position error:**
- Target may be at workspace boundary
- Try position closer to center (40, 0, 30)

## Example Usage

1. **Start application:**
   ```bash
   python main_gui.py
   ```

2. **Move to specific position:**
   - Enter: X=40, Y=0, Z=30
   - Enter: Roll=0, Pitch=0, Yaw=0
   - Click "Move"
   - Watch robot move smoothly
   - Check error metrics turn green

3. **Test random positions:**
   - Click "Random Test"
   - Robot moves to random reachable position
   - Repeat to explore workspace

4. **Emergency stop:**
   - Click "Stop" during motion
   - Robot halts immediately
   - Status shows "READY"

## Technical Details

**Update Frequencies:**
- Physics simulation: 240 Hz (4.17 ms per step)
- GUI refresh: 30 Hz (33 ms per update)
- Trajectory execution: 60 steps over ~250 ms

**Coordinate Systems:**
- Internal: meters, radians
- Display: centimeters, degrees
- Conversions handled automatically

**Joint Limits:**
- Enforced in IK computation
- Motion rejected if limits exceeded
- Logged as error in red

## Future Enhancements

- Embed PyBullet view in right panel (requires OpenGL widget)
- Add joint angle sliders for direct control
- Save/load target positions
- Trajectory recording and playback
- Neural network IK option
