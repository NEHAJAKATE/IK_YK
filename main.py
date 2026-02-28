"""
Main test runner for robotic arm IK validation
"""
import numpy as np
from simulation import RobotSimulation
from kinematics import forward_kinematics, HOME_CONFIG


def test_home_configuration():
    """Test 1: Verify home configuration matches extended arm"""
    print("\n" + "="*60)
    print("TEST 1: HOME CONFIGURATION")
    print("="*60)
    
    sim = RobotSimulation(gui=True)
    
    print(f"Home config (deg): {np.degrees(HOME_CONFIG)}")
    
    # Set to home and get FK
    sim.set_joint_angles(HOME_CONFIG)
    T = forward_kinematics(HOME_CONFIG)
    pos = T[:3, 3]
    
    print(f"End-effector position: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}] m")
    print("Visual check: Arm should be extended forward")
    print("Press Enter to continue...")
    input()
    
    sim.close()


def test_fk_ik_consistency():
    """Test 2: Random joint angles → FK → IK → FK validation"""
    print("\n" + "="*60)
    print("TEST 2: FK-IK CONSISTENCY")
    print("="*60)
    
    sim = RobotSimulation(gui=False)
    success = sim.validate_fk_ik(num_tests=20, verbose=True)
    
    if success:
        print("[OK] All tests passed")
    else:
        print("[FAIL] Some tests failed")
    
    sim.close()
    return success


def test_target_reaching():
    """Test 3: Move to specific target positions with visual feedback"""
    print("\n" + "="*60)
    print("TEST 3: TARGET REACHING")
    print("="*60)
    
    sim = RobotSimulation(gui=True)
    
    # Start at home
    sim.set_joint_angles(HOME_CONFIG)
    
    # Define test targets (position + orientation)
    targets = [
        {
            "name": "Forward reach",
            "pos": np.array([0.5, 0.0, 0.3]),
            "rot": np.eye(3)
        },
        {
            "name": "Left reach",
            "pos": np.array([0.3, 0.3, 0.2]),
            "rot": np.eye(3)
        },
        {
            "name": "Right reach",
            "pos": np.array([0.3, -0.3, 0.2]),
            "rot": np.eye(3)
        },
        {
            "name": "High reach",
            "pos": np.array([0.4, 0.0, 0.5]),
            "rot": np.eye(3)
        }
    ]
    
    for target in targets:
        print(f"\nTarget: {target['name']}")
        print(f"Position: {target['pos']}")
        
        # Show target sphere
        sim.show_target_sphere(target['pos'])
        
        # Attempt to reach
        success = sim.move_to_pose(target['pos'], target['rot'], interpolate=True)
        
        if success:
            # Verify actual position
            actual_pos, actual_rot = sim.get_end_effector_pose()
            error = np.linalg.norm(actual_pos - target['pos']) * 1000
            print(f"[OK] Reached target (error: {error:.2f} mm)")
        else:
            print("[FAIL] Failed to reach target")
        
        print("Press Enter for next target...")
        input()
    
    sim.close()


def test_workspace_exploration():
    """Test 4: Explore reachable workspace"""
    print("\n" + "="*60)
    print("TEST 4: WORKSPACE EXPLORATION")
    print("="*60)
    
    sim = RobotSimulation(gui=True)
    sim.set_joint_angles(HOME_CONFIG)
    
    print("Moving through workspace grid...")
    
    # Grid of points
    x_range = np.linspace(0.3, 0.6, 5)
    y_range = np.linspace(-0.2, 0.2, 5)
    z_range = np.linspace(0.2, 0.4, 3)
    
    reachable = 0
    total = 0
    
    for x in x_range:
        for y in y_range:
            for z in z_range:
                target_pos = np.array([x, y, z])
                target_rot = np.eye(3)
                
                sim.show_target_sphere(target_pos, radius=0.01)
                
                success = sim.move_to_pose(target_pos, target_rot, interpolate=False)
                total += 1
                if success:
                    reachable += 1
    
    print(f"\nReachability: {reachable}/{total} points ({100*reachable/total:.1f}%)")
    print("Press Enter to finish...")
    input()
    
    sim.close()


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("6-DOF ROBOTIC ARM - ANALYTICAL IK VALIDATION")
    print("="*60)
    
    tests = [
        ("Home Configuration", test_home_configuration),
        ("FK-IK Consistency", test_fk_ik_consistency),
        ("Target Reaching", test_target_reaching),
        ("Workspace Exploration", test_workspace_exploration)
    ]
    
    print("\nAvailable tests:")
    for i, (name, _) in enumerate(tests, 1):
        print(f"{i}. {name}")
    print("0. Run all tests")
    
    choice = input("\nSelect test (0-4): ").strip()
    
    if choice == "0":
        for name, test_func in tests:
            test_func()
    elif choice.isdigit() and 1 <= int(choice) <= len(tests):
        tests[int(choice) - 1][1]()
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
