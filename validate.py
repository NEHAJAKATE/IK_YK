"""
Quick validation of kinematics
"""
import numpy as np
from kinematics import forward_kinematics, HOME_CONFIG, JOINT_LIMITS
from simulation import RobotSimulation

def validate_home_config():
    """Check home configuration FK"""
    print("\n" + "="*60)
    print("HOME CONFIGURATION VALIDATION")
    print("="*60)
    
    T = forward_kinematics(HOME_CONFIG)
    pos = T[:3, 3]
    
    print(f"Home config (deg): {np.degrees(HOME_CONFIG)}")
    print(f"Expected: [0, 0, 0, 0, 0, 0]")
    print(f"\nEnd-effector position: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}] m")
    print(f"Expected: Extended forward (positive X)")
    
    if pos[0] > 0.2 and abs(pos[1]) < 0.1:
        print("[OK] Position looks correct for forward pose")
    else:
        print("[WARN] Position may not match expected forward pose")

def validate_fk_ik_roundtrip():
    """Test FK -> IK -> FK consistency using PyBullet"""
    print("\n" + "="*60)
    print("FK-IK ROUND-TRIP VALIDATION (PyBullet IK)")
    print("="*60)
    
    sim = RobotSimulation(gui=False)
    success = sim.validate_fk_ik(num_tests=10, verbose=True)
    sim.close()
    
    return success

def validate_specific_poses():
    """Test specific reachable poses"""
    print("\n" + "="*60)
    print("SPECIFIC POSE VALIDATION")
    print("="*60)
    
    sim = RobotSimulation(gui=False)
    
    test_poses = [
        ("Forward", np.array([0.4, 0.0, 0.3])),
        ("Left", np.array([0.3, 0.2, 0.3])),
        ("Right", np.array([0.3, -0.2, 0.3])),
        ("High", np.array([0.3, 0.0, 0.5]))
    ]
    
    for name, target_pos in test_poses:
        target_rot = np.eye(3)
        success = sim.move_to_pose(target_pos, target_rot, interpolate=False)
        
        if success:
            actual_pos, _ = sim.get_end_effector_pose()
            error = np.linalg.norm(actual_pos - target_pos) * 1000
            
            status = "[OK]" if error < 10.0 else "[FAIL]"
            print(f"{status} {name:10s}: target={target_pos}, error={error:.2f}mm")
        else:
            print(f"[FAIL] {name:10s}: IK failed (unreachable)")
    
    sim.close()

if __name__ == "__main__":
    print("\n" + "="*60)
    print("KINEMATICS VALIDATION")
    print("="*60)
    
    validate_home_config()
    validate_fk_ik_roundtrip()
    validate_specific_poses()
    
    print("\n" + "="*60)
    print("Validation complete. Run 'python main.py' for GUI tests.")
    print("="*60 + "\n")
