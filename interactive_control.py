"""
Real-time Interactive Control - User inputs position and orientation
"""
import numpy as np
from simulation import RobotSimulation
from kinematics import forward_kinematics, HOME_CONFIG


def parse_position(input_str):
    """Parse position string like '40 0 30' (cm) to numpy array (m)"""
    try:
        values = [float(x.strip()) / 100.0 for x in input_str.split()]  # Convert cm to m
        if len(values) != 3:
            return None
        return np.array(values)
    except:
        return None


def parse_orientation(input_str):
    """Parse orientation - supports multiple formats"""
    input_str = input_str.strip().lower()
    
    # Identity (default)
    if input_str in ['', 'identity', 'id', 'i']:
        return np.eye(3)
    
    # Euler angles (degrees): 'roll pitch yaw'
    try:
        values = [float(x.strip()) for x in input_str.split()]
        if len(values) == 3:
            roll, pitch, yaw = np.radians(values)
            # ZYX Euler angles
            cr, sr = np.cos(roll), np.sin(roll)
            cp, sp = np.cos(pitch), np.sin(pitch)
            cy, sy = np.cos(yaw), np.sin(yaw)
            
            R = np.array([
                [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                [-sp, cp*sr, cp*cr]
            ])
            return R
    except:
        pass
    
    return None


def print_current_state(sim):
    """Display current robot state"""
    q = sim.get_joint_angles()
    pos, rot = sim.get_end_effector_pose()
    
    print(f"\n{'='*60}")
    print("CURRENT STATE:")
    print(f"Joint angles (deg): {np.degrees(q).round(1)}")
    print(f"Position (cm): [{pos[0]*100:.1f}, {pos[1]*100:.1f}, {pos[2]*100:.1f}]")
    print(f"{'='*60}\n")


def main():
    print("\n" + "="*60)
    print("REAL-TIME INTERACTIVE CONTROL")
    print("="*60)
    print("\nStarting simulation...")
    
    sim = RobotSimulation(gui=True)
    sim.set_joint_angles(HOME_CONFIG)
    
    print("\nCommands:")
    print("  pos <x y z>           - Move to position (centimeters)")
    print("  ori <roll pitch yaw>  - Set orientation (degrees)")
    print("  move <x y z> [r p y]  - Move to pose (cm + deg)")
    print("  home                  - Return to home")
    print("  state                 - Show current state")
    print("  quit                  - Exit")
    
    current_orientation = np.eye(3)
    
    while True:
        try:
            cmd = input("\n> ").strip()
            
            if not cmd:
                continue
            
            parts = cmd.split(maxsplit=1)
            action = parts[0].lower()
            
            if action == 'quit':
                break
            
            elif action == 'home':
                sim.set_joint_angles(HOME_CONFIG, interpolate=True)
                print("Moved to home position")
                print_current_state(sim)
            
            elif action == 'state':
                print_current_state(sim)
            
            elif action == 'pos':
                if len(parts) < 2:
                    print("Usage: pos <x y z>")
                    continue
                
                pos = parse_position(parts[1])
                if pos is None:
                    print("Invalid position. Use: pos <x y z> (in cm)")
                    continue
                
                sim.show_target_sphere(pos)
                success = sim.move_to_pose(pos, current_orientation, interpolate=True)
                
                if success:
                    actual_pos, _ = sim.get_end_effector_pose()
                    error = np.linalg.norm(actual_pos - pos) * 1000
                    print(f"[OK] Reached target (error: {error:.2f}mm)")
                    print_current_state(sim)
                else:
                    print("[FAIL] Could not reach target (unreachable or out of limits)")
            
            elif action == 'ori':
                if len(parts) < 2:
                    print("Usage: ori <roll pitch yaw> (degrees) or 'identity'")
                    continue
                
                rot = parse_orientation(parts[1])
                if rot is None:
                    print("Invalid orientation. Use: ori <roll pitch yaw> or 'identity'")
                    continue
                
                current_orientation = rot
                print(f"[OK] Orientation set")
            
            elif action == 'move':
                if len(parts) < 2:
                    print("Usage: move <x y z> [roll pitch yaw]")
                    continue
                
                tokens = parts[1].split()
                if len(tokens) < 3:
                    print("Usage: move <x y z> [roll pitch yaw] (position in cm)")
                    continue
                
                pos = parse_position(' '.join(tokens[:3]))
                if pos is None:
                    print("Invalid position (use cm)")
                    continue
                
                if len(tokens) >= 6:
                    rot = parse_orientation(' '.join(tokens[3:6]))
                    if rot is None:
                        print("Invalid orientation, using current")
                        rot = current_orientation
                else:
                    rot = current_orientation
                
                sim.show_target_sphere(pos)
                success = sim.move_to_pose(pos, rot, interpolate=True)
                
                if success:
                    actual_pos, _ = sim.get_end_effector_pose()
                    error = np.linalg.norm(actual_pos - pos) * 1000
                    print(f"[OK] Reached target (error: {error:.2f}mm)")
                    print_current_state(sim)
                else:
                    print("[FAIL] Could not reach target")
            
            else:
                print(f"Unknown command: {action}")
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nClosing simulation...")
    sim.close()
    print("Done.")


if __name__ == "__main__":
    main()
