"""
GUI Real-time Control - Simple interface for position/orientation input
"""
import numpy as np
from simulation import RobotSimulation
from kinematics import HOME_CONFIG
import pybullet as p


def main():
    print("\n" + "="*60)
    print("GUI REAL-TIME CONTROL")
    print("="*60)
    
    sim = RobotSimulation(gui=True)
    sim.set_joint_angles(HOME_CONFIG)
    
    # Add sliders for position control (in cm)
    x_slider = p.addUserDebugParameter("Target X (cm)", -80, 80, 40)
    y_slider = p.addUserDebugParameter("Target Y (cm)", -80, 80, 0)
    z_slider = p.addUserDebugParameter("Target Z (cm)", 0, 100, 30)
    
    # Add sliders for orientation (Euler angles in degrees)
    roll_slider = p.addUserDebugParameter("Roll (deg)", -180, 180, 0)
    pitch_slider = p.addUserDebugParameter("Pitch (deg)", -180, 180, 0)
    yaw_slider = p.addUserDebugParameter("Yaw (deg)", -180, 180, 0)
    
    print("\nUse GUI sliders to control target position and orientation")
    print("Close PyBullet window to exit\n")
    
    last_values = None
    
    while True:
        try:
            # Read slider values (convert cm to m)
            x = p.readUserDebugParameter(x_slider) / 100.0
            y = p.readUserDebugParameter(y_slider) / 100.0
            z = p.readUserDebugParameter(z_slider) / 100.0
            
            roll_deg = p.readUserDebugParameter(roll_slider)
            pitch_deg = p.readUserDebugParameter(pitch_slider)
            yaw_deg = p.readUserDebugParameter(yaw_slider)
            
            current_values = (x, y, z, roll_deg, pitch_deg, yaw_deg)
            
            # Update only if values changed
            if last_values is None or current_values != last_values:
                target_pos = np.array([x, y, z])
                
                # Compute rotation matrix from Euler angles (ZYX)
                roll = np.radians(roll_deg)
                pitch = np.radians(pitch_deg)
                yaw = np.radians(yaw_deg)
                
                cr, sr = np.cos(roll), np.sin(roll)
                cp, sp = np.cos(pitch), np.sin(pitch)
                cy, sy = np.cos(yaw), np.sin(yaw)
                
                target_rot = np.array([
                    [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                    [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                    [-sp, cp*sr, cp*cr]
                ])
                
                # Show target
                sim.show_target_sphere(target_pos, radius=0.02)
                
                # Move robot
                success = sim.move_to_pose(target_pos, target_rot, interpolate=False)
                
                if success:
                    actual_pos, _ = sim.get_end_effector_pose()
                    error = np.linalg.norm(actual_pos - target_pos) * 1000
                    print(f"Target: [{x*100:.1f}, {y*100:.1f}, {z*100:.1f}]cm | Ori: [{roll_deg:.0f}, {pitch_deg:.0f}, {yaw_deg:.0f}]deg | Error: {error:.1f}mm")
                else:
                    print(f"Target: [{x*100:.1f}, {y*100:.1f}, {z*100:.1f}]cm | UNREACHABLE")
                
                last_values = current_values
            
            # Keep simulation running
            p.stepSimulation()
            
        except p.error:
            # PyBullet window closed
            break
        except KeyboardInterrupt:
            break
    
    print("\nExiting...")
    sim.close()


if __name__ == "__main__":
    main()
